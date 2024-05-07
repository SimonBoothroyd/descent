import copy

import openff.interchange
import openff.interchange.models
import openff.toolkit
import pydantic
import pytest
import smee
import smee.converters
import torch

from descent.train import AttributeConfig, ParameterConfig, Trainable, _PotentialKey


@pytest.fixture()
def mock_ff() -> smee.TensorForceField:
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-2.0.0.offxml", load_plugins=True),
        openff.toolkit.Molecule.from_smiles("CC").to_topology(),
    )

    ff, _ = smee.converters.convert_interchange(interchange)

    # check the force field matches when the tests were written.
    assert ff.potentials_by_type["vdW"].attribute_cols == (
        "scale_12",
        "scale_13",
        "scale_14",
        "scale_15",
        "cutoff",
        "switch_width",
    )

    assert ff.potentials_by_type["vdW"].parameter_cols == ("epsilon", "sigma")

    expected_vdw_ids = ["[#6X4:1]", "[#1:1]-[#6X4]"]
    vdw_ids = [key.id for key in ff.potentials_by_type["vdW"].parameter_keys]
    assert vdw_ids == expected_vdw_ids

    assert ff.potentials_by_type["Bonds"].parameter_cols == ("k", "length")

    expected_bond_ids = ["[#6X4:1]-[#6X4:2]", "[#6X4:1]-[#1:2]"]
    bond_ids = [key.id for key in ff.potentials_by_type["Bonds"].parameter_keys]
    assert bond_ids == expected_bond_ids

    return ff


@pytest.fixture()
def mock_parameter_configs(mock_ff):
    return {
        "vdW": ParameterConfig(
            cols=["epsilon", "sigma"],
            scales={"epsilon": 10.0, "sigma": 1.0},
            limits={"epsilon": (0.0, None), "sigma": (0.0, None)},
            include=[mock_ff.potentials_by_type["vdW"].parameter_keys[0]],
        ),
        "Bonds": ParameterConfig(
            cols=["length"],
            scales={"length": 1.0},
            limits={"length": (0.1, 0.7)},
            exclude=[mock_ff.potentials_by_type["Bonds"].parameter_keys[0]],
        ),
    }


@pytest.fixture()
def mock_attribute_configs():
    return {
        "vdW": AttributeConfig(
            cols=["scale_14"],
            scales={"scale_14": 0.1},
            limits={"scale_14": (0.0, None)},
        )
    }


class TestAttributeConfig:
    def test_validate_keys_scale(self):
        with pytest.raises(
            pydantic.ValidationError, match="cannot scale non-trainable parameters"
        ):
            AttributeConfig(cols=["scale_14"], scales={"scale_15": 0.1})

    def test_validate_keys_limits(self):
        with pytest.raises(
            pydantic.ValidationError, match="cannot clamp non-trainable parameters"
        ):
            AttributeConfig(cols=["scale_14"], limits={"scale_15": (0.1, 0.2)})


class TestParameterConfig:
    def test_validate_include_exclude(self):
        config = ParameterConfig(
            cols=["sigma"],
            include=[openff.interchange.models.PotentialKey(id="a")],
            exclude=[openff.interchange.models.PotentialKey(id="b")],
        )
        assert isinstance(config.include[0], _PotentialKey)
        assert isinstance(config.exclude[0], _PotentialKey)

        with pytest.raises(
            pydantic.ValidationError,
            match="cannot include and exclude the same parameter",
        ):
            ParameterConfig(
                cols=["sigma"],
                include=[openff.interchange.models.PotentialKey(id="a")],
                exclude=[openff.interchange.models.PotentialKey(id="a")],
            )


class TestTrainable:
    def test_init(self, mock_ff, mock_parameter_configs, mock_attribute_configs):
        potentials = mock_ff.potentials_by_type

        trainable = Trainable(
            mock_ff,
            parameters=mock_parameter_configs,
            attributes=mock_attribute_configs,
        )

        assert trainable._param_types == ["Bonds", "vdW"]
        assert trainable._param_shapes == [(2, 2), (2, 2)]

        assert trainable._attr_types == ["vdW"]
        assert trainable._attr_shapes == [(6,)]

        # values should be params then attrs (i.e. bond params, vdw params, vdw attrs)
        assert trainable._values.shape == (14,)
        assert torch.allclose(
            trainable._values,
            torch.cat(
                [
                    potentials["Bonds"].parameters.flatten(),
                    potentials["vdW"].parameters.flatten(),
                    potentials["vdW"].attributes.flatten(),
                ]
            ),
        )

        # bond params: k, l, k, l where only second l is unfrozen
        # vdw params: eps, sig, eps, sig where only first row is unfrozen
        # vdw attrs: only scale_14 is unfrozen
        expected_unfrozen_ids = torch.tensor([3, 4, 5, 10])
        assert (trainable._unfrozen_idxs == expected_unfrozen_ids).all()

        assert torch.allclose(
            trainable._clamp_lower,
            torch.tensor([0.1, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert torch.allclose(
            trainable._clamp_upper,
            torch.tensor([0.7, torch.inf, torch.inf, torch.inf], dtype=torch.float64),
        )
        assert torch.allclose(
            trainable._scales,
            torch.tensor([1.0, 10.0, 1.0, 0.1], dtype=torch.float64),
        )

    def test_to_values(self, mock_ff, mock_parameter_configs, mock_attribute_configs):
        potentials = mock_ff.potentials_by_type

        trainable = Trainable(
            mock_ff,
            parameters=mock_parameter_configs,
            attributes=mock_attribute_configs,
        )

        vdw_params = potentials["vdW"].parameters.flatten()
        vdw_attrs = potentials["vdW"].attributes.flatten()

        expected_values = torch.tensor(
            [
                0.7,  # length clamped
                vdw_params[0] * 10.0,  # scale eps
                vdw_params[1],  # sigma
                vdw_attrs[2] * 0.1,  # scale_14
            ]
        )
        values = trainable.to_values()

        assert values.shape == expected_values.shape
        assert torch.allclose(values, expected_values)

    def test_to_force_field_no_op(
        self, mock_ff, mock_parameter_configs, mock_attribute_configs
    ):
        mock_parameter_configs["Bonds"].limits = {"length": (0.1, None)}

        ff_initial = copy.deepcopy(mock_ff)

        trainable = Trainable(
            mock_ff,
            parameters=mock_parameter_configs,
            attributes=mock_attribute_configs,
        )

        ff = trainable.to_force_field(trainable.to_values())

        assert (
            ff.potentials_by_type["vdW"].parameters.shape
            == ff_initial.potentials_by_type["vdW"].parameters.shape
        )
        assert torch.allclose(
            ff.potentials_by_type["vdW"].parameters,
            ff_initial.potentials_by_type["vdW"].parameters,
        )

        assert (
            ff.potentials_by_type["vdW"].attributes.shape
            == ff_initial.potentials_by_type["vdW"].attributes.shape
        )
        assert torch.allclose(
            ff.potentials_by_type["vdW"].attributes,
            ff_initial.potentials_by_type["vdW"].attributes,
        )

        assert (
            ff.potentials_by_type["Bonds"].parameters.shape
            == ff_initial.potentials_by_type["Bonds"].parameters.shape
        )
        assert torch.allclose(
            ff.potentials_by_type["Bonds"].parameters,
            ff_initial.potentials_by_type["Bonds"].parameters,
        )

    def test_to_force_field_clamp(
        self, mock_ff, mock_parameter_configs, mock_attribute_configs
    ):
        ff_initial = copy.deepcopy(mock_ff)

        trainable = Trainable(
            mock_ff,
            parameters=mock_parameter_configs,
            attributes=mock_attribute_configs,
        )

        ff = trainable.to_force_field(trainable.to_values())

        expected_bond_params = ff_initial.potentials_by_type["Bonds"].parameters.clone()
        expected_bond_params[1, 1] = 0.7

        assert (
            ff.potentials_by_type["Bonds"].parameters.shape
            == expected_bond_params.shape
        )
        assert torch.allclose(
            ff.potentials_by_type["Bonds"].parameters, expected_bond_params
        )

    def test_clamp(self, mock_ff, mock_parameter_configs, mock_attribute_configs):
        potentials = mock_ff.potentials_by_type

        trainable = Trainable(
            mock_ff,
            parameters=mock_parameter_configs,
            attributes=mock_attribute_configs,
        )

        vdw_params = potentials["vdW"].parameters.flatten()
        vdw_attrs = potentials["vdW"].attributes.flatten()

        expected_values = torch.tensor([0.7, 0.0, vdw_params[1], vdw_attrs[2] * 0.1])
        values = trainable.clamp(
            torch.tensor([2.0, -1.0, vdw_params[1], vdw_attrs[2] * 0.1])
        )

        assert values.shape == expected_values.shape
        assert torch.allclose(values, expected_values)
