import pytest
from rdkit import Chem

from descent.utils.molecule import map_smiles, mol_to_smiles, unmap_smiles


@pytest.mark.parametrize(
    "input_smiles, expected_smiles, canonical",
    [
        ("OC", "[H:1][C:4]([H:2])([O:3][H:5])[H:6]", True),
        ("OC", "[O:1]([C:2]([H:4])([H:5])[H:6])[H:3]", False),
    ],
)
def test_mol_to_smiles(input_smiles, expected_smiles, canonical):
    mol = Chem.MolFromSmiles(input_smiles)
    actual_smiles = mol_to_smiles(mol, canonical)

    assert actual_smiles == expected_smiles


def test_unmap_smiles():
    smiles = "[H:1][C:4]([H:2])([O:3][H:5])[H:6]"
    unmapped_smiles = unmap_smiles(smiles)

    assert unmapped_smiles == "CO"


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("C", "[C:1]([H:2])([H:3])([H:4])[H:5]"),
        ("[CH4:1]", "[C:1]([H:2])([H:3])([H:4])[H:5]"),
        ("[Cl:1][H:2]", "[Cl:1][H:2]"),
        ("[Cl:2][H:1]", "[Cl:2][H:1]"),
        ("[Cl:2][H:2]", "[Cl:1][H:2]"),
        ("[Cl:1][H]", "[Cl:1][H:2]"),
        ("[H:1][C:4]([H:2])([O:3][H:5])[H:6]", "[H:1][C:4]([H:2])([O:3][H:5])[H:6]"),
        ("CO", "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]"),
    ],
)
def test_map_smiles(smiles, expected):
    assert map_smiles(smiles) == expected
