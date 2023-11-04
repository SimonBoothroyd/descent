import pytest
from rdkit import Chem

from descent.utils.molecule import mol_to_smiles


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
