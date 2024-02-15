"""Utilities for reporting results."""

import base64
import io
import itertools
import typing

import openff.units
import smee

if typing.TYPE_CHECKING:
    from matplotlib import pyplot
    from rdkit import Chem


DEFAULT_COLORS, DEFAULT_MARKERS = zip(
    *itertools.product(["red", "green", "blue", "black"], ["x", "o", "+", "^"])
)


def _mol_from_smiles(smiles: str) -> "Chem.Mol":
    from rdkit import Chem

    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol


def mols_to_img(*smiles: str, width: int = 400, height: int = 200) -> str:
    """Renders a set of molecules as an embeddable HTML image tag.

    Args:
        *smiles: The SMILES patterns of the molecules to render.
        width: The width of the image.
        height: The height of the image.

    Returns:
        The HTML image tag.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    assert len(smiles) > 0

    mol = _mol_from_smiles(smiles[0])

    for pattern in smiles[1:]:
        mol = Chem.CombineMols(mol, _mol_from_smiles(pattern))

    mol = Draw.PrepareMolForDrawing(mol, forceCoords=True)

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    data = base64.b64encode(drawer.GetDrawingText().encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}"></img>'


def figure_to_img(figure: "pyplot.Figure") -> str:
    """Convert a matplotlib figure to an embeddable HTML image tag.

    Args:
        figure: The figure to convert.

    Returns:
        The HTML image tag.
    """

    with io.BytesIO() as stream:
        figure.savefig(stream, format="svg")
        data = base64.b64encode(stream.getvalue()).decode()

    return f'<img src="data:image/svg+xml;base64,{data}"></img>'


def _format_unit(unit: openff.units.Unit | None) -> str:
    """Format a unit for display in a table."""

    if unit is None or unit == openff.units.unit.dimensionless:
        return ""

    return f" [{unit: ~P}]"


def _format_parameter_id(id_: typing.Any) -> str:
    """Format a parameter ID for display in a table."""

    id_str = id_ if "EP" not in id_ else id_[: id_.index("EP") + 2]
    return id_str[:60] + (id_str[60:] and "...")


def print_potential_summary(potential: smee.TensorPotential):
    """Print a summary of the potential parameters to the terminal.

    Args:
        potential: The potential.
    """
    import pandas

    parameter_rows = []

    for key, value in zip(potential.parameter_keys, potential.parameters.detach()):
        row = {"ID": _format_parameter_id(key.id)}
        row.update(
            {
                f"{col}{_format_unit(potential.parameter_units[idx])}": f"{value[idx].item():.4f}"
                for idx, col in enumerate(potential.parameter_cols)
            }
        )
        parameter_rows.append(row)

    print(f" {potential.type} ".center(88, "="), flush=True)
    print(f"fn={potential.fn}", flush=True)

    if potential.attributes is not None:
        attribute_rows = [
            {
                f"{col}{_format_unit(potential.attribute_units[idx])}": f"{potential.attributes[idx].item():.4f} "
                for idx, col in enumerate(potential.attribute_cols)
            }
        ]
        print("")
        print("attributes=", flush=True)
        print("")
        print(pandas.DataFrame(attribute_rows).to_string(index=False), flush=True)

    print("")
    print("parameters=", flush=True)
    print("")
    print(pandas.DataFrame(parameter_rows).to_string(index=False), flush=True)


def print_v_site_summary(v_sites: smee.TensorVSites):
    import pandas

    parameter_rows = []

    for key, value in zip(v_sites.keys, v_sites.parameters.detach()):
        row = {"ID": _format_parameter_id(key.id)}
        row.update(
            {
                f"{col}{_format_unit(unit)}": f"{value[idx].item():.4f}"
                for idx, (col, unit) in enumerate(v_sites.parameter_units.items())
            }
        )
        parameter_rows.append(row)

    print(" v-sites ".center(88, "="), flush=True)
    print("parameters:", flush=True)
    print(pandas.DataFrame(parameter_rows).to_string(index=False), flush=True)


def print_force_field_summary(force_field: smee.TensorForceField):
    """Print a summary of the force field parameters to the terminal.

    Args:
        force_field: The force field.
    """

    if force_field.v_sites is not None:
        print_v_site_summary(force_field.v_sites)
        print("")

    for potential in force_field.potentials:
        print_potential_summary(potential)
        print("")
