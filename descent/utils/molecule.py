import typing

if typing.TYPE_CHECKING:
    from rdkit import Chem


def mol_to_smiles(mol: "Chem.Mol", canonical: bool = True) -> str:
    """Convert a molecule to a SMILES string with atom mapping.

    Args:
        mol: The molecule to convert.
        canonical: Whether to canonicalize the atom ordering prior to assigning
            map indices.

    Returns:
        The SMILES string.
    """
    from rdkit import Chem

    mol = Chem.AddHs(mol)

    if canonical:
        order = Chem.CanonicalRankAtoms(mol, includeChirality=True)
        mol = Chem.RenumberAtoms(mol, list(order))

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    return Chem.MolToSmiles(mol)
