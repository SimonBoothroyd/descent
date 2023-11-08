"""Train against thermodynamic properties."""
import contextlib
import hashlib
import logging
import pathlib
import pickle
import typing

import numpy
import openmm.unit
import pyarrow
import pydantic
import smee.mm
import torch
from rdkit import Chem

_LOGGER = logging.getLogger(__name__)


DataType = typing.Literal["density", "hvap", "hmix"]

DATA_TYPES = typing.get_args(DataType)

DATA_SCHEMA = pyarrow.schema(
    [
        ("type", pyarrow.string()),
        ("smiles_a", pyarrow.string()),
        ("x_a", pyarrow.float64()),
        ("smiles_b", pyarrow.string()),
        ("x_b", pyarrow.float64()),
        ("temperature", pyarrow.float64()),
        ("pressure", pyarrow.float64()),
        ("value", pyarrow.float64()),
        ("std", pyarrow.float64()),
        ("units", pyarrow.string()),
        ("source", pyarrow.string()),
    ]
)

_REQUIRES_BULK_SIM = {"density": True, "hvap": True, "hmix": True}
"""Whether a bulk simulation is required for each data type."""
_REQUIRES_PURE_SIM = {"density": False, "hvap": False, "hmix": True}
"""Whether a simulation of each component is required for each data type."""
_REQUIRES_VACUUM_SIM = {"density": False, "hvap": True, "hmix": False}
"""Whether a vacuum simulation is required for each data type."""

Phase = typing.Literal["bulk", "vacuum"]
PHASES = typing.get_args(Phase)


class DataEntry(typing.TypedDict):
    """Represents a single experimental data point."""

    type: DataType
    """The type of data point."""

    smiles_a: str
    """The SMILES definition of the first component."""
    x_a: float | None
    """The mole fraction of the first component. This must be set to 1.0 if the data"""

    smiles_b: str | None
    """The SMILES definition of the second component if present."""
    x_b: float | None
    """The mole fraction of the second component if present."""

    temperature: float
    """The temperature at which the data point was measured."""
    pressure: float
    """The pressure at which the data point was measured."""

    value: float
    """The value of the data point."""
    std: float | None
    """The standard deviation of the data point if available."""
    units: str
    """The units of the data point."""

    source: str
    """The source of the data point."""


class SimulationKey(typing.NamedTuple):
    """A key used to identify a simulation."""

    smiles: tuple[str, ...]
    """The SMILES definitions of the components present in the system."""
    counts: tuple[int, ...]
    """The number of copies of each component present in the system."""

    temperature: float
    """The temperature [K] at which the simulation was run."""
    pressure: float | None
    """The pressure [atm] at which the simulation was run."""


class SimulationConfig(pydantic.BaseModel):
    """Configuration for a simulation to run."""

    max_mols: int = pydantic.Field(
        ..., description="The maximum number of molecules to simulate."
    )
    gen_coords: smee.mm.GenerateCoordsConfig = pydantic.Field(
        ..., description="Configuration for generating initial coordinates."
    )

    equilibrate: list[
        smee.mm.MinimizationConfig | smee.mm.SimulationConfig
    ] = pydantic.Field(..., description="Configuration for equilibration simulations.")

    production: smee.mm.SimulationConfig = pydantic.Field(
        ..., description="Configuration for the production simulation."
    )
    production_frequency: int = pydantic.Field(
        ..., description="The frequency at which to write frames during production."
    )


_SystemDict = dict[SimulationKey, smee.TensorSystem]


def create_dataset(*rows: DataEntry) -> pyarrow.Table:
    """Create a dataset from a list of existing data points.

    Args:
        rows: The data points to create the dataset from.

    Returns:
        The created dataset.
    """

    for row in rows:
        row["smiles_a"] = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles_a"]))

        if row["smiles_b"] is None:
            continue

        row["smiles_b"] = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles_b"]))

    # TODO: validate rows
    return pyarrow.Table.from_pylist([*rows], schema=DATA_SCHEMA)


def extract_smiles(dataset: pyarrow.Table) -> list[str]:
    """Return a list of unique SMILES strings in the dataset.

    Args:
        dataset: The dataset to extract the SMILES strings from.

    Returns:
        The unique SMILES strings with full atom mapping.
    """
    smiles_a = dataset["smiles_a"].drop_null().unique().to_pylist()
    smiles_b = dataset["smiles_b"].drop_null().unique().to_pylist()

    smiles_unique = sorted({*smiles_a, *smiles_b})
    return smiles_unique


def _convert_entry_to_system(
    entry: DataEntry, topologies: dict[str, smee.TensorTopology], max_mols: int
) -> tuple[SimulationKey, smee.TensorSystem]:
    """Convert a data entry into a system ready to simulate.

    Args:
        entry: The data entry to convert.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        max_mols: The maximum number of molecules to simulate.

    Returns:
        The system and its associated key.
    """
    smiles_a = entry["smiles_a"]
    smiles_b = entry["smiles_b"]

    fraction_a = 0.0 if entry["x_a"] is None else entry["x_a"]
    fraction_b = 0.0 if entry["x_b"] is None else entry["x_b"]

    assert numpy.isclose(fraction_a + fraction_b, 1.0)

    n_copies_a = int(max_mols * fraction_a)
    n_copies_b = int(max_mols * fraction_b)

    smiles = [smiles_a]

    system_topologies = [topologies[smiles_a]]
    n_copies = [n_copies_a]

    if n_copies_b > 0:
        smiles.append(smiles_b)

        system_topologies.append(topologies[smiles_b])
        n_copies.append(n_copies_b)

    key = SimulationKey(
        tuple(smiles), tuple(n_copies), entry["temperature"], entry["pressure"]
    )
    system = smee.TensorSystem(system_topologies, n_copies, True)

    return key, system


def _bulk_config(temperature: float, pressure: float) -> SimulationConfig:
    """Return a default simulation configuration for simulations of the bulk phase.

    Args:
        temperature: The temperature [K] at which to run the simulation.
        pressure: The pressure [atm] at which to run the simulation.

    Returns:
        The default simulation configuration.
    """
    temperature = temperature * openmm.unit.kelvin
    pressure = pressure * openmm.unit.atmosphere

    return SimulationConfig(
        max_mols=256,
        gen_coords=smee.mm.GenerateCoordsConfig(),
        equilibrate=[
            smee.mm.MinimizationConfig(),
            # short NVT equilibration simulation
            smee.mm.SimulationConfig(
                temperature=temperature,
                pressure=None,
                n_steps=50000,
                timestep=1.0 * openmm.unit.femtosecond,
            ),
            # short NPT equilibration simulation
            smee.mm.SimulationConfig(
                temperature=temperature,
                pressure=pressure,
                n_steps=50000,
                timestep=1.0 * openmm.unit.femtosecond,
            ),
        ],
        production=smee.mm.SimulationConfig(
            temperature=temperature,
            pressure=pressure,
            n_steps=500000,
            timestep=2.0 * openmm.unit.femtosecond,
        ),
        production_frequency=1000,
    )


def _vacuum_config(temperature: float, pressure: float | None) -> SimulationConfig:
    """Return a default simulation configuration for simulations of the vacuum phase.

    Args:
        temperature: The temperature [K] at which to run the simulation.
        pressure: The pressure [atm] at which to run the simulation.

    Returns:
        The default simulation configuration.
    """
    temperature = temperature * openmm.unit.kelvin
    assert pressure is None

    return SimulationConfig(
        max_mols=1,
        gen_coords=smee.mm.GenerateCoordsConfig(),
        equilibrate=[
            smee.mm.MinimizationConfig(),
            smee.mm.SimulationConfig(
                temperature=temperature,
                pressure=None,
                n_steps=50000,
                timestep=1.0 * openmm.unit.femtosecond,
            ),
        ],
        production=smee.mm.SimulationConfig(
            temperature=temperature,
            pressure=None,
            n_steps=1000000,
            timestep=1.0 * openmm.unit.femtosecond,
        ),
        production_frequency=500,
    )


def default_config(
    phase: Phase, temperature: float, pressure: float | None
) -> SimulationConfig:
    """Return a default simulation configuration for the specified phase.

    Args:
        phase: The phase to return the default configuration for.
        temperature: The temperature [K] at which to run the simulation.
        pressure: The pressure [atm] at which to run the simulation.

    Returns:
        The default simulation configuration.
    """

    if phase.lower() == "bulk":
        return _bulk_config(temperature, pressure)
    elif phase.lower() == "vacuum":
        return _vacuum_config(temperature, pressure)
    else:
        raise NotImplementedError(phase)


def _plan_simulations(
    entries: list[DataEntry], topologies: dict[str, smee.TensorTopology]
) -> tuple[dict[Phase, _SystemDict], list[dict[str, SimulationKey]]]:
    """Plan the simulations required to compute the properties in a dataset.

    Args:
        entries: The entries in the dataset.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.

    Returns:
        The systems to simulate and the simulations required to compute each property.
    """
    systems_per_phase: dict[Phase, _SystemDict] = {phase: {} for phase in PHASES}
    simulations_per_entry = []

    for entry in entries:
        data_type = entry["type"].lower()

        if data_type not in DATA_TYPES:
            raise NotImplementedError(data_type)

        required_sims: dict[str, SimulationKey] = {}

        bulk_config = default_config("bulk", entry["temperature"], entry["pressure"])
        max_mols = bulk_config.max_mols

        if _REQUIRES_BULK_SIM[data_type]:
            key, system = _convert_entry_to_system(entry, topologies, max_mols)

            systems_per_phase["bulk"][key] = system
            required_sims["bulk"] = key

        if _REQUIRES_PURE_SIM[data_type]:
            for i, smiles in enumerate((entry["smiles_a"], entry["smiles_b"])):
                key = SimulationKey(
                    (smiles,), (max_mols,), entry["temperature"], entry["pressure"]
                )
                system = smee.TensorSystem([topologies[smiles]], [max_mols], True)

                systems_per_phase["bulk"][key] = system
                required_sims[f"bulk_{i}"] = key

        if _REQUIRES_VACUUM_SIM[data_type]:
            assert entry["smiles_b"] is None, "vacuum sims only support pure systems"

            system = smee.TensorSystem([topologies[entry["smiles_a"]]], [1], False)
            key = SimulationKey((entry["smiles_a"],), (1,), entry["temperature"], None)

            systems_per_phase["vacuum"][key] = system
            required_sims["vacuum"] = key

        simulations_per_entry.append(required_sims)

    return systems_per_phase, simulations_per_entry


def _simulate(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    config: SimulationConfig,
    output_path: pathlib.Path,
):
    """Simulate a system.

    Args:
        system: The system to simulate.
        force_field: The force field to use.
        config: The simulation configuration to use.
        output_path: The path at which to write the simulation trajectory.
    """
    coords, box_vectors = smee.mm.generate_system_coords(
        system, force_field, config.gen_coords
    )

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * config.production.temperature)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output:
        reporter = smee.mm.TensorReporter(
            output, config.production_frequency, beta, config.production.pressure
        )
        smee.mm.simulate(
            system,
            force_field,
            coords,
            box_vectors,
            config.equilibrate,
            config.production,
            [reporter],
        )


def _compute_averages(
    phase: Phase,
    key: SimulationKey,
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
    cached_dir: pathlib.Path | None,
) -> dict[str, torch.Tensor]:
    traj_hash = hashlib.sha256(pickle.dumps(key)).hexdigest()
    traj_name = f"{phase}-{traj_hash}-frames.msgpack"

    cached_path = None if cached_dir is None else cached_dir / traj_name

    temperature = key.temperature * openmm.unit.kelvin
    pressure = None if key.pressure is None else key.pressure * openmm.unit.atmospheres

    if cached_path is not None and cached_path.exists():
        with contextlib.suppress(smee.mm.NotEnoughSamplesError):
            return smee.mm.reweight_ensemble_averages(
                system, force_field, cached_path, temperature, pressure
            )

    if cached_path is not None:
        _LOGGER.debug(f"unable to re-weight {key}: data exists={cached_path.exists()}")

    output_path = output_dir / traj_name

    config = default_config(phase, key.temperature, key.pressure)
    _simulate(system, force_field, config, output_path)

    return smee.mm.compute_ensemble_averages(
        system, force_field, output_path, temperature, pressure
    )


def _predict_density(
    entry: DataEntry, averages: dict[str, torch.Tensor]
) -> torch.Tensor:
    assert entry["units"] == "g/mL"
    return averages["density"]


def _predict_hvap(
    entry: DataEntry,
    averages_bulk: dict[str, torch.Tensor],
    averages_vacuum: dict[str, torch.Tensor],
    system_bulk: smee.TensorSystem,
) -> torch.Tensor:
    assert entry["units"] == "kcal/mol"

    temperature = entry["temperature"] * openmm.unit.kelvin

    potential_bulk = averages_bulk["potential_energy"] / sum(system_bulk.n_copies)
    potential_vacuum = averages_vacuum["potential_energy"]

    rt = (temperature * openmm.unit.MOLAR_GAS_CONSTANT_R).value_in_unit(
        openmm.unit.kilocalorie_per_mole
    )
    return potential_vacuum - potential_bulk + rt


def _predict_hmix(
    entry: DataEntry,
    averages_mix: dict[str, torch.Tensor],
    averages_0: dict[str, torch.Tensor],
    averages_1: dict[str, torch.Tensor],
    system_mix: smee.TensorSystem,
    system_0: smee.TensorSystem,
    system_1: smee.TensorSystem,
) -> torch.Tensor:
    assert entry["units"] == "kcal/mol"

    x_0 = system_mix.n_copies[0] / sum(system_mix.n_copies)
    x_1 = 1.0 - x_0

    enthalpy_mix = averages_mix["enthalpy"] / sum(system_mix.n_copies)

    enthalpy_0 = averages_0["enthalpy"] / sum(system_0.n_copies)
    enthalpy_1 = averages_1["enthalpy"] / sum(system_1.n_copies)

    return enthalpy_mix - x_0 * enthalpy_0 - x_1 * enthalpy_1


def _predict(
    entry: DataEntry,
    keys: dict[str, SimulationKey],
    averages: dict[Phase, dict[SimulationKey, dict[str, torch.Tensor]]],
    systems: dict[Phase, dict[SimulationKey, smee.TensorSystem]],
):
    if entry["type"] == "density":
        value = _predict_density(entry, averages["bulk"][keys["bulk"]])
    elif entry["type"] == "hvap":
        value = _predict_hvap(
            entry,
            averages["bulk"][keys["bulk"]],
            averages["vacuum"][keys["vacuum"]],
            systems["bulk"][keys["bulk"]],
        )
    elif entry["type"] == "hmix":
        value = _predict_hmix(
            entry,
            averages["bulk"][keys["bulk"]],
            averages["bulk"][keys["bulk_0"]],
            averages["bulk"][keys["bulk_1"]],
            systems["bulk"][keys["bulk"]],
            systems["bulk"][keys["bulk_0"]],
            systems["bulk"][keys["bulk_1"]],
        )
    else:
        raise NotImplementedError(entry["type"])

    return value


def predict(
    dataset: pyarrow.Table,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    output_dir: pathlib.Path,
    cached_dir: pathlib.Path | None = None,
    per_type_scales: dict[DataType, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the properties in a dataset using molecular simulation, or by reweighting
    previous simulation data.

    Args:
        dataset: The dataset to predict the properties of.
        force_field: The force field to use.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        output_dir: The directory to write the simulation trajectories to.
        cached_dir: The (optional) directory to read cached simulation trajectories
            from.
        per_type_scales: The scale factor to apply to each data type. A default of 1.0
            will be used for any data type not specified.
    """

    entries: list[DataEntry] = dataset.to_pylist()

    required_simulations, entry_to_simulation = _plan_simulations(entries, topologies)
    averages = {
        phase: {
            key: _compute_averages(
                phase, key, system, force_field, output_dir, cached_dir
            )
            for key, system in systems.items()
        }
        for phase, systems in required_simulations.items()
    }

    predicted = []
    reference = []

    per_type_scales = per_type_scales if per_type_scales is not None else {}

    for entry, keys in zip(entries, entry_to_simulation):
        value = _predict(entry, keys, averages, required_simulations)

        predicted.append(value * per_type_scales.get(entry["type"], 1.0))
        reference.append(
            torch.tensor(entry["value"]) * per_type_scales.get(entry["type"], 1.0)
        )

    predicted = torch.stack(predicted)
    reference = torch.stack(reference).to(predicted.device)

    return reference, predicted
