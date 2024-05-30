import numpy
import openmm.unit
import pytest
import smee.mm
import torch
import uncertainties.unumpy

import descent.utils.dataset
from descent.targets.thermo import (
    DataEntry,
    SimulationKey,
    _compute_observables,
    _convert_entry_to_system,
    _Observables,
    _plan_simulations,
    _predict,
    _simulate,
    create_dataset,
    create_from_evaluator,
    default_closure,
    default_config,
    extract_smiles,
    predict,
)


@pytest.fixture
def mock_density_pure() -> DataEntry:
    return {
        "type": "density",
        "smiles_a": "CO",
        "x_a": 1.0,
        "smiles_b": None,
        "x_b": None,
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 0.785,
        "std": 0.001,
        "units": "g/mL",
        "source": None,
    }


@pytest.fixture
def mock_density_binary() -> DataEntry:
    return {
        "type": "density",
        "smiles_a": "CCO",
        "x_a": 0.5,
        "smiles_b": "CO",
        "x_b": 0.5,
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 0.9,
        "std": 0.002,
        "units": "g/mL",
        "source": None,
    }


@pytest.fixture
def mock_hvap() -> DataEntry:
    return {
        "type": "hvap",
        "smiles_a": "CCCC",
        "x_a": 1.0,
        "smiles_b": None,
        "x_b": None,
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 1.234,
        "std": 0.004,
        "units": "kcal/mol",
        "source": None,
    }


@pytest.fixture
def mock_hmix() -> DataEntry:
    return {
        "type": "hmix",
        "smiles_a": "CCO",
        "x_a": 0.5,
        "smiles_b": "CO",
        "x_b": 0.5,
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 0.4321,
        "std": 0.0025,
        "units": "kcal/mol",
        "source": None,
    }


def test_create_dataset(mock_density_pure, mock_density_binary):
    expected_entries = [mock_density_pure, mock_density_binary]

    dataset = create_dataset(*expected_entries)
    assert len(dataset) == 2

    entries = list(descent.utils.dataset.iter_dataset(dataset))

    assert entries == pytest.approx(expected_entries)


def test_extract_smiles(mock_density_pure, mock_density_binary):
    dataset = create_dataset(mock_density_pure, mock_density_binary)
    smiles = extract_smiles(dataset)

    expected_smiles = [
        "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
        "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
    ]
    assert smiles == expected_smiles


def test_convert_entry_to_system_pure(mock_density_pure, mocker):
    topology = mocker.Mock()
    topologies = {"CO": topology}

    n_mols = 123

    key, system = _convert_entry_to_system(mock_density_pure, topologies, n_mols)

    assert key == (
        ("CO",),
        (123,),
        mock_density_pure["temperature"],
        mock_density_pure["pressure"],
    )

    assert system.topologies == [topology]
    assert system.n_copies == [n_mols]
    assert system.is_periodic is True


def test_convert_entry_to_system_binary(mock_density_binary, mocker):
    topology_a = mocker.Mock()
    topology_b = mocker.Mock()
    topologies = {"CO": topology_a, "CCO": topology_b}

    n_mols = 128

    key, system = _convert_entry_to_system(mock_density_binary, topologies, n_mols)

    assert key == (
        ("CCO", "CO"),
        (n_mols // 2, n_mols // 2),
        mock_density_binary["temperature"],
        mock_density_binary["pressure"],
    )

    assert system.topologies == [topology_b, topology_a]
    assert system.n_copies == [n_mols // 2, n_mols // 2]
    assert system.is_periodic is True


@pytest.mark.parametrize(
    "phase, pressure, expected_n_mols", [("bulk", 1.23, 256), ("vacuum", None, 1)]
)
def test_default_config(phase, pressure, expected_n_mols):
    expected_temperature = 298.15

    config = default_config(phase, expected_temperature, pressure)

    assert config.max_mols == expected_n_mols

    assert (
        config.production.temperature.value_in_unit(openmm.unit.kelvin)
        == expected_temperature
    )

    if pressure is None:
        assert config.production.pressure is None
    else:
        assert (
            config.production.pressure.value_in_unit(openmm.unit.atmosphere) == pressure
        )


def test_plan_simulations(
    mock_density_pure, mock_density_binary, mock_hvap, mock_hmix, mocker
):
    topology_co = mocker.Mock()
    topology_cco = mocker.Mock()
    topology_cccc = mocker.Mock()

    topologies = {"CO": topology_co, "CCO": topology_cco, "CCCC": topology_cccc}

    required_simulations, entry_to_simulation = _plan_simulations(
        [mock_density_pure, mock_density_binary, mock_hvap, mock_hmix], topologies
    )

    assert sorted(required_simulations) == ["bulk", "vacuum"]

    expected_vacuum_key = SimulationKey(("CCCC",), (1,), mock_hvap["temperature"], None)
    assert sorted(required_simulations["vacuum"]) == [expected_vacuum_key]
    assert required_simulations["vacuum"][expected_vacuum_key].n_copies == [1]
    assert required_simulations["vacuum"][expected_vacuum_key].topologies == [
        topology_cccc
    ]

    expected_cccc_key = SimulationKey(
        ("CCCC",),
        (256,),
        mock_hvap["temperature"],
        mock_hvap["pressure"],
    )
    expected_co_key = SimulationKey(
        ("CO",),
        (256,),
        mock_density_pure["temperature"],
        mock_density_pure["pressure"],
    )
    expected_cco_key = SimulationKey(
        ("CCO",),
        (256,),
        mock_density_binary["temperature"],
        mock_density_binary["pressure"],
    )
    expected_cco_co_key = SimulationKey(
        ("CCO", "CO"),
        (128, 128),
        mock_density_binary["temperature"],
        mock_density_binary["pressure"],
    )

    expected_bulk_keys = [
        expected_cccc_key,
        expected_co_key,
        expected_cco_key,
        expected_cco_co_key,
    ]

    assert sorted(required_simulations["bulk"]) == sorted(expected_bulk_keys)

    assert required_simulations["bulk"][expected_cccc_key].n_copies == [256]
    assert required_simulations["bulk"][expected_cccc_key].topologies == [topology_cccc]

    assert required_simulations["bulk"][expected_cco_key].n_copies == [256]
    assert required_simulations["bulk"][expected_cco_key].topologies == [topology_cco]

    assert required_simulations["bulk"][expected_co_key].n_copies == [256]
    assert required_simulations["bulk"][expected_co_key].topologies == [topology_co]

    assert required_simulations["bulk"][expected_cco_co_key].n_copies == [128, 128]
    assert required_simulations["bulk"][expected_cco_co_key].topologies == [
        topology_cco,
        topology_co,
    ]

    assert entry_to_simulation == [
        {"bulk": expected_co_key},
        {"bulk": expected_cco_co_key},
        {"bulk": expected_cccc_key, "vacuum": expected_vacuum_key},
        {
            "bulk": expected_cco_co_key,
            "bulk_0": expected_cco_key,
            "bulk_1": expected_co_key,
        },
    ]


def test_simulation(tmp_cwd, mocker):
    coords = numpy.zeros((1, 3)) * openmm.unit.angstrom
    box_vectors = numpy.eye(3) * openmm.unit.angstrom

    expected_temperature = 298.15
    config = default_config("bulk", expected_temperature, 1.0)

    mock_system = mocker.MagicMock()
    mock_ff = mocker.MagicMock()

    mock_gen_coords = mocker.patch(
        "smee.mm.generate_system_coords",
        autospec=True,
        return_value=(coords, box_vectors),
    )
    mock_simulate = mocker.patch("smee.mm.simulate", autospec=True)

    spied_reporter = mocker.spy(smee.mm, "TensorReporter")

    expected_output = tmp_cwd / "frames.msgpack"
    _simulate(mock_system, mock_ff, config, expected_output)

    mock_gen_coords.assert_called_once_with(mock_system, mock_ff, config.gen_coords)

    mock_simulate.assert_called_once_with(
        mock_system,
        mock_ff,
        coords,
        box_vectors,
        config.equilibrate,
        config.production,
        [mocker.ANY],
        False,
    )
    assert expected_output.exists()

    expected_beta = 1.0 / (
        openmm.unit.MOLAR_GAS_CONSTANT_R * expected_temperature * openmm.unit.kelvin
    )

    spied_reporter.assert_called_once_with(
        mocker.ANY,
        config.production_frequency,
        expected_beta,
        config.production.pressure,
    )


def test_compute_observables_reweighted(tmp_cwd, mocker):
    mock_result = {"density": torch.randn(1)}
    mock_reweight = mocker.patch(
        "smee.mm.reweight_ensemble_averages", autospec=True, return_value=mock_result
    )

    expected_hash = "1234567890abcdef"

    mock_hash = mocker.MagicMock()
    mock_hash.hexdigest.return_value = expected_hash

    mocker.patch("hashlib.sha256", autospec=True, return_value=mock_hash)

    phase = "vacuum"
    key = SimulationKey(("CCCC",), (1,), 298.15, None)

    mock_system = mocker.Mock()
    mock_ff = mocker.Mock()

    cached_dir = tmp_cwd / "cached"
    cached_dir.mkdir()

    expected_path = cached_dir / f"{phase}-{expected_hash}-frames.msgpack"
    expected_path.touch()

    result = _compute_observables(phase, key, mock_system, mock_ff, tmp_cwd, cached_dir)
    assert result.mean == mock_result
    assert {*result.std} == {*result.mean}

    mock_reweight.assert_called_once_with(
        mock_system, mock_ff, expected_path, 298.15 * openmm.unit.kelvin, None
    )


def test_compute_observables_simulated(tmp_cwd, mocker):
    mock_result = {"density": torch.randn(1)}, {"density": torch.randn(1)}
    mocker.patch(
        "smee.mm.reweight_ensemble_averages",
        autospec=True,
        side_effect=smee.mm._ops.NotEnoughSamplesError(),
    )
    mock_simulate = mocker.patch("descent.targets.thermo._simulate", autospec=True)
    mock_compute = mocker.patch(
        "smee.mm.compute_ensemble_averages", autospec=True, return_value=mock_result
    )

    expected_hash = "1234567890abcdef"

    mock_hash = mocker.MagicMock()
    mock_hash.hexdigest.return_value = expected_hash

    mocker.patch("hashlib.sha256", autospec=True, return_value=mock_hash)

    phase = "vacuum"
    key = SimulationKey(("CCCC",), (1,), 298.15, None)

    mock_system = mocker.Mock()
    mock_ff = mocker.Mock()

    cached_dir = tmp_cwd / "cached"
    cached_dir.mkdir()
    (cached_dir / f"{phase}-{expected_hash}-frames.msgpack").touch()

    expected_path = tmp_cwd / f"{phase}-{expected_hash}-frames.msgpack"
    expected_path.touch()

    result = _compute_observables(phase, key, mock_system, mock_ff, tmp_cwd, cached_dir)
    assert result == mock_result

    mock_simulate.assert_called_once_with(
        mock_system, mock_ff, mocker.ANY, expected_path
    )
    mock_compute.assert_called_once_with(
        mock_system, mock_ff, expected_path, 298.15 * openmm.unit.kelvin, None
    )


def test_predict_density(mock_density_pure, mocker):
    topologies = {"CO": mocker.Mock()}
    key, system = _convert_entry_to_system(mock_density_pure, topologies, 123)

    expected_result = torch.randn(1)
    expected_std = torch.randn(1)

    observables = {
        "bulk": {
            key: _Observables({"density": expected_result}, {"density": expected_std})
        }
    }
    systems = {"bulk": {key: system}}

    result, std = _predict(mock_density_pure, {"bulk": key}, observables, systems)
    assert result == expected_result
    assert std == expected_std


def test_predict_hvap(mock_hvap, mocker):
    topologies = {"CCCC": mocker.Mock()}

    n_mols = 123

    key_bulk, system_bulk = _convert_entry_to_system(mock_hvap, topologies, n_mols)
    key_vaccum = SimulationKey(("CCCC",), (1,), mock_hvap["temperature"], None)

    system_vacuum = smee.TensorSystem([topologies["CCCC"]], [1], False)

    potential_bulk = torch.tensor([7.0])
    potential_bulk_std = torch.randn(1).abs()

    potential_vacuum = torch.tensor([3.0])
    potential_vacuum_std = torch.randn(1).abs()

    averages = {
        "bulk": {
            key_bulk: _Observables(
                {"potential_energy": potential_bulk},
                {"potential_energy": potential_bulk_std},
            )
        },
        "vacuum": {
            key_vaccum: _Observables(
                {"potential_energy": potential_vacuum},
                {"potential_energy": potential_vacuum_std},
            )
        },
    }
    systems = {"bulk": {key_bulk: system_bulk}, "vacuum": {key_vaccum: system_vacuum}}
    keys = {"bulk": key_bulk, "vacuum": key_vaccum}

    rt = (
        mock_hvap["temperature"] * openmm.unit.kelvin * openmm.unit.MOLAR_GAS_CONSTANT_R
    ).value_in_unit(openmm.unit.kilocalorie_per_mole)

    expected = (
        uncertainties.ufloat(potential_vacuum, potential_vacuum_std)
        - uncertainties.ufloat(potential_bulk, potential_bulk_std) / n_mols
        + rt
    )

    result, std = _predict(mock_hvap, keys, averages, systems)
    assert result == pytest.approx(expected.n)
    assert std == pytest.approx(expected.s)


def test_predict_hmix(mock_hmix, mocker):
    topologies = {"CO": mocker.Mock(), "CCO": mocker.Mock()}

    n_mols = 100

    key_bulk, system_bulk = _convert_entry_to_system(mock_hmix, topologies, n_mols)
    key_0 = SimulationKey(
        ("CCO",), (n_mols,), mock_hmix["temperature"], mock_hmix["pressure"]
    )
    key_1 = SimulationKey(
        ("CO",), (n_mols,), mock_hmix["temperature"], mock_hmix["pressure"]
    )

    system_0 = smee.TensorSystem([topologies["CCO"]], [n_mols], False)
    system_1 = smee.TensorSystem([topologies["CO"]], [n_mols], False)

    enthalpy_bulk = torch.tensor([16.0])
    enthalpy_bulk_std = torch.randn(1).abs()
    enthalpy_0 = torch.tensor([4.0])
    enthalpy_0_std = torch.randn(1).abs()
    enthalpy_1 = torch.tensor([3.0])
    enthalpy_1_std = torch.randn(1).abs()

    averages = {
        "bulk": {
            key_bulk: _Observables(
                {"enthalpy": enthalpy_bulk}, {"enthalpy": enthalpy_bulk_std}
            ),
            key_0: _Observables({"enthalpy": enthalpy_0}, {"enthalpy": enthalpy_0_std}),
            key_1: _Observables({"enthalpy": enthalpy_1}, {"enthalpy": enthalpy_1_std}),
        },
    }
    systems = {"bulk": {key_bulk: system_bulk, key_0: system_0, key_1: system_1}}
    keys = {"bulk": key_bulk, "bulk_0": key_0, "bulk_1": key_1}

    expected = (
        uncertainties.ufloat(enthalpy_bulk, enthalpy_bulk_std) / n_mols
        - 0.5 * uncertainties.ufloat(enthalpy_0, enthalpy_0_std) / n_mols
        - 0.5 * uncertainties.ufloat(enthalpy_1, enthalpy_1_std) / n_mols
    )

    result, std = _predict(mock_hmix, keys, averages, systems)
    assert result == pytest.approx(expected.n)
    assert std == pytest.approx(expected.s)


def test_predict(tmp_cwd, mock_density_pure, mocker):
    dataset = create_dataset(mock_density_pure)

    mock_topologies = {"[C:1]([O:2][H:6])([H:3])([H:4])[H:5]": mocker.Mock()}
    mock_ff = mocker.Mock()

    mock_density = torch.tensor(123.0)
    mock_density_std = torch.tensor(0.456)

    mock_compute = mocker.patch(
        "descent.targets.thermo._compute_observables",
        autospec=True,
        return_value=_Observables(
            {"density": mock_density}, {"density": mock_density_std}
        ),
    )

    mock_scale = 3.0

    y_ref, y_ref_std, y_pred, y_pred_std = predict(
        dataset,
        mock_ff,
        mock_topologies,
        tmp_cwd,
        None,
        {"density": mock_scale},
        verbose=True,
    )

    mock_compute.assert_called_once_with(
        "bulk",
        SimulationKey(
            ("[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",),
            (256,),
            mock_density_pure["temperature"],
            mock_density_pure["pressure"],
        ),
        mocker.ANY,
        mock_ff,
        tmp_cwd,
        None,
    )

    expected_y_ref = torch.tensor([mock_density_pure["value"] * mock_scale])
    expected_y_ref_std = torch.tensor([mock_density_pure["std"] * mock_scale])

    expected_y_pred = torch.tensor([mock_density * mock_scale])
    expected_y_pred_std = torch.tensor([mock_density_std * mock_scale])

    assert y_ref.shape == expected_y_ref.shape
    assert torch.allclose(y_ref, expected_y_ref)
    assert y_ref_std.shape == expected_y_ref_std.shape
    assert torch.allclose(y_ref_std, expected_y_ref_std)

    assert y_pred.shape == expected_y_pred.shape
    assert torch.allclose(y_pred, expected_y_pred)
    assert y_pred_std.shape == expected_y_pred_std.shape
    assert torch.allclose(y_pred_std, expected_y_pred_std)


def test_default_closure(tmp_cwd, mock_density_pure, mocker):
    dataset = create_dataset(mock_density_pure)

    mock_x = torch.tensor([2.0], requires_grad=True)

    mock_y_pred = torch.tensor([3.0, 4.0]) * mock_x
    mock_y_ref = torch.Tensor([-1.23, 4.56])

    mocker.patch(
        "descent.targets.thermo.predict",
        autospec=True,
        return_value=(mock_y_ref, None, mock_y_pred, None),
    )
    mock_topologies = {mock_density_pure["smiles_a"]: mocker.MagicMock()}
    mock_trainable = mocker.MagicMock()

    closure_fn = default_closure(mock_trainable, mock_topologies, dataset, None)

    expected_loss = (mock_y_pred - mock_y_ref).pow(2).sum()

    loss, grad, hess = closure_fn(mock_x, compute_gradient=True, compute_hessian=True)

    assert torch.isclose(loss, expected_loss)
    assert grad.shape == mock_x.shape
    assert hess.shape == (1, 1)


def test_create_from_evaluator(data_dir):
    dataset = create_from_evaluator(dataset_file=data_dir / "evaluator_mock.json")

    entries = list(descent.utils.dataset.iter_dataset(dataset))
    expected = {
        "smiles_a": "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
        "x_a": 0.48268,
        "smiles_b": "[O:1]([H:2])[H:3]",
        "x_b": 0.51732,
        "temperature": 298.15,
        "pressure": 0.999753269183321,
        "value": 0.99,
        "std": 0.000505,
        "units": "g/mL",
        "source": "mock",
        "type": "density",
    }
    assert entries[0] == expected


def test_unsupported_property(data_dir):
    with pytest.raises(KeyError):
        _ = create_from_evaluator(
            dataset_file=data_dir / "missing_property_evaluator.json"
        )
