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


def unmap_smiles(smiles: str) -> str:
    """Remove atom mapping from a SMILES string.

    Args:
        smiles: The SMILES string to unmap.

    Returns:
        The unmapped SMILES string.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


def map_smiles(smiles: str) -> str:
    """Add atom mapping to a SMILES string.

    Notes:
        Fully mapped SMILES strings are returned as-is.

    Args:
        smiles: The SMILES string to add map indices to.

    Returns:
        The mapped SMILES string.
    """
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, params))

    map_idxs = sorted(atom.GetAtomMapNum() for atom in mol.GetAtoms())

    if map_idxs == list(range(1, len(map_idxs) + 1)):
        return smiles

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)

    return Chem.MolToSmiles(mol)
