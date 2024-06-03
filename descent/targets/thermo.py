"""Train against thermodynamic properties."""

import contextlib
import hashlib
import logging
import pathlib
import pickle
import typing

import datasets
import datasets.table
import numpy
import openmm.unit
import pyarrow
import pydantic
import smee.mm
import smee.utils
import torch

import descent.optim
import descent.utils.dataset
import descent.utils.loss
import descent.utils.molecule

if typing.TYPE_CHECKING:
    import descent.train


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

    apply_hmr: bool = pydantic.Field(
        False, description="Whether to apply hydrogen mass repartitioning."
    )

    equilibrate: list[smee.mm.MinimizationConfig | smee.mm.SimulationConfig] = (
        pydantic.Field(..., description="Configuration for equilibration simulations.")
    )

    production: smee.mm.SimulationConfig = pydantic.Field(
        ..., description="Configuration for the production simulation."
    )
    production_frequency: int = pydantic.Field(
        ..., description="The frequency at which to write frames during production."
    )


class _Observables(typing.NamedTuple):
    """Ensemble averages of the observables computed from a simulation."""

    mean: dict[str, torch.Tensor]
    """The mean value of each observable with ``shape=()``."""
    std: dict[str, torch.Tensor]
    """The standard deviation of each observable with ``shape=()``."""


_SystemDict = dict[SimulationKey, smee.TensorSystem]


def create_dataset(*rows: DataEntry) -> datasets.Dataset:
    """Create a dataset from a list of existing data points.

    Args:
        rows: The data points to create the dataset from.

    Returns:
        The created dataset.
    """

    for row in rows:
        row["smiles_a"] = descent.utils.molecule.map_smiles(row["smiles_a"])

        if row["smiles_b"] is None:
            continue

        row["smiles_b"] = descent.utils.molecule.map_smiles(row["smiles_b"])

    # TODO: validate rows
    table = pyarrow.Table.from_pylist([*rows], schema=DATA_SCHEMA)

    dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
    return dataset


def create_from_evaluator(dataset_file: pathlib.Path) -> datasets.Dataset:
    """
    Create a dataset from an evaluator PhysicalPropertyDataSet

    Args:
        dataset_file: The path to the evaluator dataset

    Returns:
        The created dataset
    """
    import json

    from openff.units import unit

    _evaluator_to_prop = {
        "openff.evaluator.properties.density.Density": "density",
        "openff.evaluator.properties.enthalpy.EnthalpyOfMixing": "hmix",
        "openff.evaluator.properties.enthalpy.EnthalpyOfVaporization": "hvap",
    }
    _prop_units = {"density": "g/mL", "hmix": "kcal/mol", "hvap": "kcal/mol"}

    properties: list[DataEntry] = []
    property_data = json.load(dataset_file.open())

    for phys_prop in property_data["properties"]:
        try:
            prop_type = _evaluator_to_prop[phys_prop["@type"]]
        except KeyError:
            raise KeyError(f"{phys_prop['@type']} not currently supported.") from None

        smiles_and_role = [
            (comp["smiles"], comp["smiles"] + "{" + comp["role"]["value"] + "}")
            for comp in phys_prop["substance"]["components"]
        ]
        smiles_a, role_a = smiles_and_role[0]
        x_a = phys_prop["substance"]["amounts"][role_a][0]["value"]
        if len(smiles_and_role) == 1:
            smiles_b, x_b = None, None
        elif:
            smiles_b, role_b = smiles_and_role[1]
            x_b = phys_prop["substance"]["amounts"][role_b][0]["value"]
        else:
            raise NotImplementedError("up to binary mixtures are currently supported")

        temp_unit = getattr(
            unit, phys_prop["thermodynamic_state"]["temperature"]["unit"]
        )
        temp = phys_prop["thermodynamic_state"]["temperature"]["value"] * temp_unit
        pressure_unit = getattr(
            unit, phys_prop["thermodynamic_state"]["pressure"]["unit"]
        )
        pressure = phys_prop["thermodynamic_state"]["pressure"]["value"] * pressure_unit
        value = phys_prop["value"]["value"] * getattr(unit, phys_prop["value"]["unit"])
        std = phys_prop["uncertainty"]["value"] * getattr(
            unit, phys_prop["uncertainty"]["unit"]
        )
        default_units = getattr(unit, _prop_units[prop_type])
        prop = {
            "type": prop_type,
            "smiles_a": smiles_a,
            "x_a": x_a,
            "smiles_b": smiles_b,
            "x_b": x_b,
            "temperature": temp.to(unit.kelvin).m,
            "pressure": pressure.to(unit.atm).m,
            "value": value.to(default_units).m,
            "units": _prop_units[prop_type],
            "std": std.to(default_units).m,
            "source": phys_prop["source"]["doi"],
        }
        properties.append(prop)

    return create_dataset(*properties)


def extract_smiles(dataset: datasets.Dataset) -> list[str]:
    """Return a list of unique SMILES strings in the dataset.

    Args:
        dataset: The dataset to extract the SMILES strings from.

    Returns:
        The unique SMILES strings with full atom mapping.
    """
    smiles_a = {smiles for smiles in dataset.unique("smiles_a") if smiles is not None}
    smiles_b = {smiles for smiles in dataset.unique("smiles_b") if smiles is not None}

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
            config.apply_hmr,
        )


def _compute_observables(
    phase: Phase,
    key: SimulationKey,
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
    cached_dir: pathlib.Path | None,
) -> _Observables:
    traj_hash = hashlib.sha256(pickle.dumps(key)).hexdigest()
    traj_name = f"{phase}-{traj_hash}-frames.msgpack"

    cached_path = None if cached_dir is None else cached_dir / traj_name

    temperature = key.temperature * openmm.unit.kelvin
    pressure = None if key.pressure is None else key.pressure * openmm.unit.atmospheres

    if cached_path is not None and cached_path.exists():
        with contextlib.suppress(smee.mm.NotEnoughSamplesError):
            means = smee.mm.reweight_ensemble_averages(
                system, force_field, cached_path, temperature, pressure
            )
            stds = {key: smee.utils.tensor_like(torch.nan, means[key]) for key in means}

            return _Observables(means, stds)

    if cached_path is not None:
        _LOGGER.debug(f"unable to re-weight {key}: data exists={cached_path.exists()}")

    output_path = output_dir / traj_name

    config = default_config(phase, key.temperature, key.pressure)
    _simulate(system, force_field, config, output_path)

    return _Observables(
        *smee.mm.compute_ensemble_averages(
            system, force_field, output_path, temperature, pressure
        )
    )


def _predict_density(
    entry: DataEntry, observables: _Observables
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert entry["units"] == "g/mL"
    return observables.mean["density"], observables.std["density"]


def _predict_hvap(
    entry: DataEntry,
    observables_bulk: _Observables,
    observables_vacuum: _Observables,
    system_bulk: smee.TensorSystem,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert entry["units"] == "kcal/mol"

    temperature = entry["temperature"] * openmm.unit.kelvin
    n_mols = sum(system_bulk.n_copies)

    potential_bulk = observables_bulk.mean["potential_energy"] / n_mols
    potential_bulk_std = observables_bulk.std["potential_energy"] / n_mols

    potential_vacuum = observables_vacuum.mean["potential_energy"]
    potential_vacuum_std = observables_vacuum.std["potential_energy"]

    rt = (temperature * openmm.unit.MOLAR_GAS_CONSTANT_R).value_in_unit(
        openmm.unit.kilocalorie_per_mole
    )

    value = potential_vacuum - potential_bulk + rt
    std = torch.sqrt(potential_vacuum_std**2 + potential_bulk_std**2)

    return value, std


def _predict_hmix(
    entry: DataEntry,
    observables_mix: _Observables,
    observables_0: _Observables,
    observables_1: _Observables,
    system_mix: smee.TensorSystem,
    system_0: smee.TensorSystem,
    system_1: smee.TensorSystem,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert entry["units"] == "kcal/mol"

    n_mols_mix = sum(system_mix.n_copies)
    n_mols_0 = sum(system_0.n_copies)
    n_mols_1 = sum(system_1.n_copies)

    x_0 = system_mix.n_copies[0] / n_mols_mix
    x_1 = 1.0 - x_0

    enthalpy_mix = observables_mix.mean["enthalpy"] / n_mols_mix
    enthalpy_mix_std = observables_mix.std["enthalpy"] / n_mols_mix

    enthalpy_0 = observables_0.mean["enthalpy"] / n_mols_0
    enthalpy_0_std = observables_0.std["enthalpy"] / n_mols_0
    enthalpy_1 = observables_1.mean["enthalpy"] / n_mols_1
    enthalpy_1_std = observables_1.std["enthalpy"] / n_mols_1

    value = enthalpy_mix - x_0 * enthalpy_0 - x_1 * enthalpy_1
    std = torch.sqrt(
        enthalpy_mix_std**2 + x_0**2 * enthalpy_0_std**2 + x_1**2 * enthalpy_1_std**2
    )

    return value, std


def _predict(
    entry: DataEntry,
    keys: dict[str, SimulationKey],
    observables: dict[Phase, dict[SimulationKey, _Observables]],
    systems: dict[Phase, dict[SimulationKey, smee.TensorSystem]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if entry["type"] == "density":
        value = _predict_density(entry, observables["bulk"][keys["bulk"]])
    elif entry["type"] == "hvap":
        value = _predict_hvap(
            entry,
            observables["bulk"][keys["bulk"]],
            observables["vacuum"][keys["vacuum"]],
            systems["bulk"][keys["bulk"]],
        )
    elif entry["type"] == "hmix":
        value = _predict_hmix(
            entry,
            observables["bulk"][keys["bulk"]],
            observables["bulk"][keys["bulk_0"]],
            observables["bulk"][keys["bulk_1"]],
            systems["bulk"][keys["bulk"]],
            systems["bulk"][keys["bulk_0"]],
            systems["bulk"][keys["bulk_1"]],
        )
    else:
        raise NotImplementedError(entry["type"])

    return value


def predict(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    output_dir: pathlib.Path,
    cached_dir: pathlib.Path | None = None,
    per_type_scales: dict[DataType, float] | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        verbose: Whether to log additional information.
    """

    entries: list[DataEntry] = [*descent.utils.dataset.iter_dataset(dataset)]

    required_simulations, entry_to_simulation = _plan_simulations(entries, topologies)
    observables = {
        phase: {
            key: _compute_observables(
                phase, key, system, force_field, output_dir, cached_dir
            )
            for key, system in systems.items()
        }
        for phase, systems in required_simulations.items()
    }

    predicted = []
    predicted_std = []
    reference = []
    reference_std = []

    verbose_rows = []

    per_type_scales = per_type_scales if per_type_scales is not None else {}

    for entry, keys in zip(entries, entry_to_simulation, strict=True):
        value, std = _predict(entry, keys, observables, required_simulations)

        type_scale = per_type_scales.get(entry["type"], 1.0)

        predicted.append(value * type_scale)
        predicted_std.append(torch.nan if std is None else std * abs(type_scale))

        reference.append(entry["value"] * type_scale)
        reference_std.append(
            torch.nan if entry["std"] is None else entry["std"] * abs(type_scale)
        )

        if verbose:
            std_ref = "" if entry["std"] is None else f" ± {float(entry['std']):.3f}"

            verbose_rows.append(
                {
                    "type": f'{entry["type"]} [{entry["units"]}]',
                    "smiles_a": descent.utils.molecule.unmap_smiles(entry["smiles_a"]),
                    "smiles_b": (
                        ""
                        if entry["smiles_b"] is None
                        else descent.utils.molecule.unmap_smiles(entry["smiles_b"])
                    ),
                    "pred": f"{float(value):.3f} ± {float(std):.3f}",
                    "ref": f"{float(entry['value']):.3f}{std_ref}",
                }
            )

    if verbose:
        import pandas

        _LOGGER.info(f"predicted {len(entries)} properties")
        _LOGGER.info("\n" + pandas.DataFrame(verbose_rows).to_string(index=False))

    predicted = torch.stack(predicted)
    predicted_std = torch.stack(predicted_std)

    reference = smee.utils.tensor_like(reference, predicted)
    reference_std = smee.utils.tensor_like(reference_std, predicted_std)

    return reference, reference_std, predicted, predicted_std


def default_closure(
    trainable: "descent.train.Trainable",
    topologies: dict[str, smee.TensorTopology],
    dataset: datasets.Dataset,
    per_type_scales: dict[DataType, float] | None = None,
    verbose: bool = False,
) -> descent.optim.ClosureFn:
    """Return a default closure function for training against thermodynamic
    properties.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        per_type_scales: The scale factor to apply to each data type.
        verbose: Whether to log additional information about predictions.

    Returns:
        The default closure function.
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ):
        force_field = trainable.to_force_field(x)

        y_ref, _, y_pred, _ = descent.targets.thermo.predict(
            dataset,
            force_field,
            topologies,
            pathlib.Path.cwd(),
            None,
            per_type_scales,
            verbose,
        )
        loss, gradient, hessian = ((y_pred - y_ref) ** 2).sum(), None, None

        if compute_hessian:
            hessian = descent.utils.loss.approximate_hessian(x, y_pred)
        if compute_gradient:
            gradient = torch.autograd.grad(loss, x, retain_graph=True)[0].detach()

        return loss.detach(), gradient, hessian

    return closure_fn
