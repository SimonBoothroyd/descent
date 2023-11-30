"""Train against dimer energies."""
import pathlib
import typing

import datasets
import datasets.table
import pyarrow
import smee
import smee.utils
import torch
import tqdm

import descent.utils.dataset
import descent.utils.molecule
import descent.utils.reporting

if typing.TYPE_CHECKING:
    import pandas


EnergyFn = typing.Callable[
    ["pandas.DataFrame", tuple[str, ...], torch.Tensor], torch.Tensor
]


DATA_SCHEMA = pyarrow.schema(
    [
        ("smiles_a", pyarrow.string()),
        ("smiles_b", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("source", pyarrow.string()),
    ]
)


class Dimer(typing.TypedDict):
    """Represents a single experimental data point."""

    smiles_a: str
    smiles_b: str

    coords: torch.Tensor
    energy: torch.Tensor

    source: str


def create_dataset(dimers: list[Dimer]) -> datasets.Dataset:
    """Create a dataset from a list of existing dimers.

    Args:
        dimers: The dimers to create the dataset from.

    Returns:
        The created dataset.
    """

    table = pyarrow.Table.from_pylist(
        [
            {
                "smiles_a": dimer["smiles_a"],
                "smiles_b": dimer["smiles_b"],
                "coords": torch.tensor(dimer["coords"]).flatten().tolist(),
                "energy": torch.tensor(dimer["energy"]).flatten().tolist(),
                "source": dimer["source"],
            }
            for dimer in dimers
        ],
        schema=DATA_SCHEMA,
    )
    # TODO: validate rows
    dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
    dataset.set_format("torch", columns=['coords', 'energy'], dytpe=torch.float64)

    return dataset


def create_from_des(
    data_dir: pathlib.Path,
    energy_fn: EnergyFn,
) -> datasets.Dataset:
    """Create a dataset from a DESXXX dimer set.

    Args:
        data_dir: The path to the DESXXX directory.
        energy_fn: A function which computes the reference energy of a dimer. This
            should take as input a pandas DataFrame containing the metadata for a
            given group, a tuple of geometry IDs, and a tensor of coordinates with
            ``shape=(n_dimers, n_atoms, 3)``. It should return a tensor of energies
            with ``shape=(n_dimers,)`` and units of [kcal/mol].

    Returns:
        The created dataset.
    """
    import pandas
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    metadata = pandas.read_csv(data_dir / f"{data_dir.name}.csv", index_col=False)

    system_ids = metadata["system_id"].unique()
    dimers: list[Dimer] = []

    for system_id in tqdm.tqdm(system_ids, desc="loading dimers"):
        system_data = metadata[metadata["system_id"] == system_id]

        group_ids = metadata[metadata["system_id"] == system_id]["group_id"].unique()

        for group_id in group_ids:
            group_data = system_data[system_data["group_id"] == group_id]
            group_orig = group_data["group_orig"].unique()[0]

            geometry_ids = tuple(group_data["geom_id"].values)

            dimer_example = Chem.MolFromMolFile(
                f"{data_dir}/geometries/{system_id}/DES{group_orig}_{geometry_ids[0]}.mol",
                removeHs=False,
            )
            mol_a, mol_b = Chem.GetMolFrags(dimer_example, asMols=True)

            smiles_a = descent.utils.molecule.mol_to_smiles(mol_a, False)
            smiles_b = descent.utils.molecule.mol_to_smiles(mol_b, False)

            source = (
                f"{data_dir.name} system={system_id} orig={group_orig} group={group_id}"
            )

            coords_raw = [
                Chem.MolFromMolFile(
                    f"{data_dir}/geometries/{system_id}/DES{group_orig}_{geometry_id}.mol",
                    removeHs=False,
                )
                .GetConformer()
                .GetPositions()
                .tolist()
                for geometry_id in geometry_ids
            ]

            coords = torch.tensor(coords_raw, dtype=torch.float64)
            energy = energy_fn(group_data, geometry_ids, coords)

            dimer = {
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "coords": coords,
                "energy": energy,
                "source": source,
            }
            dimers.append(dimer)

    RDLogger.EnableLog("rdApp.*")

    return create_dataset(dimers)


def extract_smiles(dataset: datasets.Dataset) -> list[str]:
    """Return a list of unique SMILES strings in the dataset.

    Args:
        dataset: The dataset to extract the SMILES strings from.

    Returns:
        The list of unique SMILES strings.
    """

    smiles_a = dataset.unique("smiles_a")
    smiles_b = dataset.unique("smiles_b")

    return sorted({*smiles_a, *smiles_b})


def compute_dimer_energy(
    topology_a: smee.TensorTopology,
    topology_b: smee.TensorTopology,
    force_field: smee.TensorForceField,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Compute the energy of a dimer in a series of conformers.

    Args:
        topology_a: The topology of the first monomer.
        topology_b: The topology of the second monomer.
        force_field: The force field to use.
        coords: The coordinates of the dimer with ``shape=(n_dimers, n_atoms, 3)``.

    Returns:
        The energy [kcal/mol] of the dimer in each conformer.
    """
    dimer = smee.TensorSystem([topology_a, topology_b], [1, 1], False)

    coords_a = coords[:, : topology_a.n_atoms, :]

    if topology_a.v_sites is not None:
        coords_a = smee.geometry.add_v_site_coords(
            topology_a.v_sites, coords_a, force_field
        )

    coords_b = coords[:, topology_a.n_atoms :, :]

    if topology_b.v_sites is not None:
        coords_b = smee.geometry.add_v_site_coords(
            topology_b.v_sites, coords_b, force_field
        )

    coords = torch.cat([coords_a, coords_b], dim=1)

    energy_dimer = smee.compute_energy(dimer, force_field, coords)

    energy_a = smee.compute_energy(topology_a, force_field, coords_a)
    energy_b = smee.compute_energy(topology_b, force_field, coords_b)

    return energy_dimer - energy_a - energy_b


def _predict(
    dimer: Dimer,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the energies of a single dimer in multiple conformations.

    Args:
        dimer: The dimer to predict the energies of.
        force_field: The force field to use.
        topologies: The topologies of each monomer. Each key should be a fully
            mapped SMILES string.

    Returns:
        The reference and predicted energies [kcal/mol] with ``shape=(n_confs,)``.
    """

    n_coords = len(dimer["energy"])

    coords_flat = smee.utils.tensor_like(
        dimer["coords"], force_field.potentials[0].parameters
    )
    coords = coords_flat.reshape(n_coords, -1, 3)

    predicted = compute_dimer_energy(
        topologies[dimer["smiles_a"]],
        topologies[dimer["smiles_b"]],
        force_field,
        coords,
    )
    reference = smee.utils.tensor_like(dimer["energy"], predicted)

    return reference, predicted


def predict(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the energies of each dimer in the dataset.

    Args:
        dataset: The dataset to predict the energies of.
        force_field: The force field to use.
        topologies: The topologies of each monomer. Each key should be a fully
            mapped SMILES string.

    Returns:
        The reference and predicted energies [kcal/mol] of each dimer, each with
        ``shape=(n_dimers * n_conf_per_dimer,)``.
    """

    reference, predicted = zip(
        *[
            _predict(dimer, force_field, topologies)
            for dimer in descent.utils.dataset.iter_dataset(dataset)
        ]
    )
    return torch.cat(reference), torch.cat(predicted)


def _plot_energies(energies: dict[str, torch.Tensor]) -> str:
    from matplotlib import pyplot

    figure, axis = pyplot.subplots(1, 1, figsize=(4.0, 4.0))

    for i, (k, v) in enumerate(energies.items()):
        axis.plot(
            v.cpu().detach().numpy(),
            label=k,
            linestyle="none",
            marker=descent.utils.reporting.DEFAULT_MARKERS[i],
            color=descent.utils.reporting.DEFAULT_COLORS[i],
        )

    axis.set_xlabel("Idx")
    axis.set_ylabel("Energy [kcal / mol]")

    axis.legend()

    figure.tight_layout()
    img = descent.utils.reporting.figure_to_img(figure)

    pyplot.close(figure)

    return img


def report(
    dataset: datasets.Dataset,
    force_fields: dict[str, smee.TensorForceField],
    topologies: dict[str, smee.TensorTopology],
    output_path: pathlib.Path,
):
    """Generate a report comparing the predicted and reference energies of each dimer.

    Args:
        dataset: The dataset to generate the report for.
        force_fields: The force fields to use to predict the energies.
        topologies: The topologies of each monomer. Each key should be a fully
            mapped SMILES string.
        output_path: The path to write the report to.
    """
    import pandas

    rows = []

    delta_sqr_total = {
        force_field_name: torch.zeros(1) for force_field_name in force_fields
    }
    delta_sqr_count = 0

    for dimer in descent.utils.dataset.iter_dataset(dataset):
        energies = {"ref": dimer["energy"]}
        energies.update(
            (force_field_name, _predict(dimer, force_field, topologies)[1])
            for force_field_name, force_field in force_fields.items()
        )

        mol_img = descent.utils.reporting.mols_to_img(
            dimer["smiles_a"], dimer["smiles_b"]
        )
        data_row = {"Dimer": mol_img, "Energy [kcal/mol]": _plot_energies(energies)}

        for force_field_name in force_fields:
            delta_sqr = ((energies["ref"] - energies[force_field_name]) ** 2).sum()
            delta_sqr_total[force_field_name] += delta_sqr

            rmse = torch.sqrt(delta_sqr / len(energies["ref"]))
            data_row[f"RMSE {force_field_name}"] = rmse.item()

        data_row["Source"] = dimer["source"]

        delta_sqr_count += len(energies["ref"])

        rows.append(data_row)

    rmse_total_rows = [
        {
            "Force Field": force_field_name,
            "RMSE [kcal/mol]": torch.sqrt(
                delta_sqr_total[force_field_name].sum() / delta_sqr_count
            ).item(),
        }
        for force_field_name in force_fields
    ]

    import bokeh.models.widgets.tables
    import panel

    data_full = pandas.DataFrame(rows)
    data_stats = pandas.DataFrame(rmse_total_rows)

    rmse_format = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")

    formatters_stats = {
        col: rmse_format for col in data_stats.columns if col.startswith("RMSE")
    }
    formatters_full = {
        **{col: "html" for col in ["Dimer", "Energy [kcal/mol]"]},
        **{col: rmse_format for col in data_full.columns if col.startswith("RMSE")},
    }

    layout = panel.Column(
        "## Statistics",
        panel.widgets.Tabulator(
            pandas.DataFrame(rmse_total_rows),
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters_stats,
            configuration={"columnDefaults": {"headerSort": False}},
        ),
        "## Energies",
        panel.widgets.Tabulator(
            data_full,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters_full,
            configuration={"rowHeight": 400},
        ),
        sizing_mode="stretch_width",
        scroll=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout.save(output_path, title="Dimers", embed=True)
