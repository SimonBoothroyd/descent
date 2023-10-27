from matplotlib import pyplot
from rdkit import Chem

from descent.utils.reporting import _mol_from_smiles, figure_to_img, mols_to_img


def test_mol_from_smiles():
    mol = _mol_from_smiles("[H:2][C:1]([H:3])([H:4])[O:5][H:6]")
    assert Chem.MolToSmiles(mol) == "CO"


def test_mols_to_img():
    img = mols_to_img("CO", "CC")
    assert img.startswith('<img src="data:image/svg+xml;base64,')
    assert img.endswith('"></img>')


def test_figure_to_img():
    figure = pyplot.figure()
    img = figure_to_img(figure)
    pyplot.close(figure)

    assert img.startswith('<img src="data:image/svg+xml;base64,')
    assert img.endswith('"></img>')
