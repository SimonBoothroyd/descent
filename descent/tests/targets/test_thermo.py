import numpy
import openmm.unit
import pytest
import smee.mm
import torch

from descent.targets.thermo import (
    DataEntry,
    SimulationKey,
    _compute_averages,
    _convert_entry_to_system,
    _plan_simulations,
    _predict,
    _simulate,
    create_dataset,
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
    expected_data_entries = [mock_density_pure, mock_density_binary]

    dataset = create_dataset(*expected_data_entries)
    assert len(dataset) == 2

    data_entries = dataset.to_pylist()

    assert data_entries == pytest.approx(expected_data_entries)


def test_extract_smiles(mock_density_pure, mock_density_binary):
    dataset = create_dataset(mock_density_pure, mock_density_binary)
    smiles = extract_smiles(dataset)

    expected_smiles = [
        "[H:1][C:4]([H:3])([O:5][H:2])[C:9]([H:6])([H:7])[H:8]",
        "[H:1][O:3][C:6]([H:2])([H:4])[H:5]",
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

    mock_gen_coords.assert_called_once_with(mock_system, config.gen_coords)

    mock_simulate.assert_called_once_with(
        mock_system,
        mock_ff,
        coords,
        box_vectors,
        config.equilibrate,
        config.production,
        [mocker.ANY],
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


def test_compute_averages_reweighted(tmp_cwd, mocker):
    mock_result = mocker.Mock()
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

    result = _compute_averages(phase, key, mock_system, mock_ff, tmp_cwd, cached_dir)
    assert result == mock_result

    mock_reweight.assert_called_once_with(
        mock_system, mock_ff, expected_path, 298.15 * openmm.unit.kelvin, None
    )


def test_compute_averages_simulated(tmp_cwd, mocker):
    mock_result = mocker.Mock()
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

    result = _compute_averages(phase, key, mock_system, mock_ff, tmp_cwd, cached_dir)
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

    expected_result = mocker.Mock()

    averages = {"bulk": {key: {"density": expected_result}}}
    systems = {"bulk": {key: system}}

    result = _predict(mock_density_pure, {"bulk": key}, averages, systems)
    assert result == expected_result


def test_predict_hvap(mock_hvap, mocker):
    topologies = {"CCCC": mocker.Mock()}

    n_mols = 123

    key_bulk, system_bulk = _convert_entry_to_system(mock_hvap, topologies, n_mols)
    key_vaccum = SimulationKey(("CCCC",), (1,), mock_hvap["temperature"], None)

    system_vacuum = smee.TensorSystem([topologies["CCCC"]], [1], False)

    potential_bulk = torch.tensor([7.0])
    potential_vacuum = torch.tensor([3.0])

    averages = {
        "bulk": {key_bulk: {"potential_energy": potential_bulk}},
        "vacuum": {key_vaccum: {"potential_energy": potential_vacuum}},
    }
    systems = {"bulk": {key_bulk: system_bulk}, "vacuum": {key_vaccum: system_vacuum}}
    keys = {"bulk": key_bulk, "vacuum": key_vaccum}

    rt = (
        mock_hvap["temperature"] * openmm.unit.kelvin * openmm.unit.MOLAR_GAS_CONSTANT_R
    ).value_in_unit(openmm.unit.kilocalorie_per_mole)

    expected = potential_vacuum - potential_bulk / n_mols + rt

    result = _predict(mock_hvap, keys, averages, systems)
    assert result == pytest.approx(expected)


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
    enthalpy_0 = torch.tensor([4.0])
    enthalpy_1 = torch.tensor([3.0])

    averages = {
        "bulk": {
            key_bulk: {"enthalpy": enthalpy_bulk},
            key_0: {"enthalpy": enthalpy_0},
            key_1: {"enthalpy": enthalpy_1},
        },
    }
    systems = {"bulk": {key_bulk: system_bulk, key_0: system_0, key_1: system_1}}
    keys = {"bulk": key_bulk, "bulk_0": key_0, "bulk_1": key_1}

    expected = (
        enthalpy_bulk / n_mols - 0.5 * enthalpy_0 / n_mols - 0.5 * enthalpy_1 / n_mols
    )

    result = _predict(mock_hmix, keys, averages, systems)
    assert result == pytest.approx(expected)


def test_predict(tmp_cwd, mock_density_pure, mocker):
    dataset = create_dataset(mock_density_pure)

    mock_topologies = {"CO": mocker.Mock()}
    mock_ff = mocker.Mock()

    mock_density = torch.tensor(123.0)

    mock_compute = mocker.patch(
        "descent.targets.thermo._compute_averages",
        autospec=True,
        return_value={"density": mock_density},
    )

    mock_scale = 3.0

    y_ref, y_pred = predict(
        dataset, mock_ff, mock_topologies, tmp_cwd, None, {"density": mock_scale}
    )

    mock_compute.assert_called_once_with(
        "bulk",
        SimulationKey(
            ("CO",),
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
    expected_y_pred = torch.tensor([mock_density * mock_scale])

    assert y_ref.shape == expected_y_ref.shape
    assert torch.allclose(y_ref, expected_y_ref)

    assert y_pred.shape == expected_y_pred.shape
    assert torch.allclose(y_pred, expected_y_pred)