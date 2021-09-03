import pytest
import torch
from openff.interchange.components.interchange import Interchange
from openff.interchange.models import PotentialKey
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from smirnoffee.smirnoff import vectorize_system

from descent.models import ParameterizationModel
from descent.models.smirnoff import SMIRNOFFModel


@pytest.fixture()
def mock_force_field() -> ForceField:

    from simtk import unit as simtk_unit

    force_field = ForceField()

    parameters = {
        "Bonds": [
            {
                "smirks": "[#1:1]-[#9:2]",
                "k": 1.0 * simtk_unit.kilojoules_per_mole / simtk_unit.angstrom ** 2,
                "length": 2.0 * simtk_unit.angstrom,
            },
            {
                "smirks": "[#1:1]-[#17:2]",
                "k": 3.0 * simtk_unit.kilojoules_per_mole / simtk_unit.angstrom ** 2,
                "length": 4.0 * simtk_unit.angstrom,
            },
            {
                "smirks": "[#1:1]-[#8:2]",
                "k": 5.0 * simtk_unit.kilojoules_per_mole / simtk_unit.angstrom ** 2,
                "length": 6.0 * simtk_unit.angstrom,
            },
        ],
        "Angles": [
            {
                "smirks": "[#1:1]-[#8:2]-[#1:3]",
                "k": 1.0 * simtk_unit.kilojoules_per_mole / simtk_unit.degrees ** 2,
                "angle": 2.0 * simtk_unit.degrees,
            }
        ],
    }

    for handler_type, parameter_dicts in parameters.items():

        handler = force_field.get_parameter_handler(handler_type)

        for parameter_dict in parameter_dicts:
            handler.add_parameter(parameter_dict)

    return force_field


def test_model_matches_protocol():
    assert issubclass(SMIRNOFFModel, ParameterizationModel)


def test_model_init(mock_force_field):

    expected_parameter_ids = [
        ("Bonds", "[#1:1]-[#17:2]", "length"),
        ("Angles", "[#1:1]-[#8:2]-[#1:3]", "angle"),
        ("Bonds", "[#1:1]-[#9:2]", "k"),
    ]

    model = SMIRNOFFModel(expected_parameter_ids, initial_force_field=mock_force_field)

    assert model._initial_force_field == mock_force_field

    assert {*model._parameter_delta_ids} == {"Bonds", "Angles"}
    assert model._parameter_delta_ids["Bonds"] == [
        (PotentialKey(id="[#1:1]-[#17:2]", associated_handler="Bonds"), "length"),
        (PotentialKey(id="[#1:1]-[#9:2]", associated_handler="Bonds"), "k"),
    ]
    assert model._parameter_delta_ids["Angles"] == [
        (PotentialKey(id="[#1:1]-[#8:2]-[#1:3]", associated_handler="Angles"), "angle")
    ]

    assert torch.allclose(model._parameter_delta_indices["Bonds"], torch.tensor([0, 1]))
    assert torch.allclose(model._parameter_delta_indices["Angles"], torch.tensor([2]))

    assert model.parameter_delta.shape == (3,)


def test_model_parameter_delta_ids():

    input_parameter_ids = [
        ("Bonds", "[#1:1]-[#17:2]", "length"),
        ("Angles", "[#1:1]-[#8:2]-[#1:3]", "angle"),
        ("Bonds", "[#1:1]-[#9:2]", "k"),
    ]
    expected_parameter_ids = tuple(
        (
            handler_type,
            PotentialKey(id=smirks, associated_handler=handler_type),
            attribute,
        )
        for handler_type, smirks, attribute in [
            ("Bonds", "[#1:1]-[#17:2]", "length"),
            ("Bonds", "[#1:1]-[#9:2]", "k"),
            ("Angles", "[#1:1]-[#8:2]-[#1:3]", "angle"),
        ]
    )

    model = SMIRNOFFModel(input_parameter_ids, None)
    assert model.parameter_delta_ids == expected_parameter_ids


def test_model_forward_empty_input():

    model = SMIRNOFFModel([("Bonds", "[#1:1]-[#17:2]", "length")], None)
    assert model.forward({}) == {}


def test_model_forward(mock_force_field):

    molecule = Molecule.from_smiles("[H]Cl")
    system = Interchange.from_smirnoff(mock_force_field, molecule.to_topology())

    model = SMIRNOFFModel([("Bonds", "[#1:1]-[#17:2]", "length")], None)
    model.parameter_delta = torch.nn.Parameter(
        model.parameter_delta + torch.tensor([1.0]), requires_grad=True
    )

    input_system = vectorize_system(system)
    output_system = model.forward(input_system)

    assert torch.isclose(
        output_system[("Bonds", "k/2*(r-length)**2")][1][0, 1], torch.tensor(5.0)
    )


def test_model_forward_fixed_handler(mock_force_field):
    """Test that forward works for the case where a system contains a handler that
    contains no parameters being trained."""

    molecule = Molecule.from_smiles("O")
    system = Interchange.from_smirnoff(mock_force_field, molecule.to_topology())

    model = SMIRNOFFModel([("Bonds", "[#1:1]-[#17:2]", "length")], None)

    input_system = vectorize_system(system)
    output_system = model.forward(input_system)

    assert ("Angles", "k/2*(theta-angle)**2") in output_system

    assert len(output_system[("Angles", "k/2*(theta-angle)**2")][0]) == 1
    assert len(output_system[("Angles", "k/2*(theta-angle)**2")][1]) == 1
    assert len(output_system[("Angles", "k/2*(theta-angle)**2")][2]) == 1
