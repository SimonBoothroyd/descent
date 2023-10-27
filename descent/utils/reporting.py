"""Utilities for reporting results."""
import base64
import io
import itertools
import typing

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
