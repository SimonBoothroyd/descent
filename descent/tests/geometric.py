from collections import defaultdict
from typing import Dict, Tuple

import torch
from openff.toolkit.topology import Molecule


def _geometric_internal_coordinate_to_indices(internal_coordinate):
    """A utility method for converting a ``geometric`` internal coordinate into
    a tuple of atom indices.

    Args:
        internal_coordinate: The internal coordinate to convert.

    Returns:
        A tuple of the relevant atom indices.
    """

    from geometric.internal import Angle, Dihedral, Distance, OutOfPlane

    if isinstance(internal_coordinate, Distance):
        indices = (internal_coordinate.a, internal_coordinate.b)
    elif isinstance(internal_coordinate, Angle):
        indices = (internal_coordinate.a, internal_coordinate.b, internal_coordinate.c)
    elif isinstance(internal_coordinate, (Dihedral, OutOfPlane)):
        indices = (
            internal_coordinate.a,
            internal_coordinate.b,
            internal_coordinate.c,
            internal_coordinate.d,
        )
    else:
        raise NotImplementedError()

    if indices[-1] > indices[0]:
        indices = tuple(reversed(indices))

    return indices


def geometric_hessian(
    molecule: Molecule,
    conformer: torch.Tensor,
    internal_coordinates_indices: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """A helper method to project a set of gradients and hessians into internal
    coordinates using ``geomTRIC``.

    Args:
        molecule: The molecule of interest
        conformer: The conformer of the molecule with units of [A] and shape=(n_atoms, 3)
        internal_coordinates_indices: The indices of the atoms involved in each type
            of internal coordinate.

    Returns:
        The projected gradients and hessians.
    """

    from geometric.internal import Angle, Dihedral, Distance, OutOfPlane
    from geometric.internal import PrimitiveInternalCoordinates as GeometricPRIC
    from geometric.internal import (
        RotationA,
        RotationB,
        RotationC,
        TranslationX,
        TranslationY,
        TranslationZ,
    )
    from geometric.molecule import Molecule as GeometricMolecule

    geometric_molecule = GeometricMolecule()
    geometric_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.element.symbol for atom in molecule.atoms],
        "bonds": [(bond.atom1_index, bond.atom2_index) for bond in molecule.bonds],
        "name": molecule.name,
        "xyzs": [conformer.detach().numpy()],
    }

    geometric_coordinates = GeometricPRIC(geometric_molecule)

    geometric_coordinates.Internals = [
        internal
        for internal in geometric_coordinates.Internals
        if not isinstance(
            internal,
            (TranslationX, TranslationY, TranslationZ, RotationA, RotationB, RotationC),
        )
    ]

    # We need to re-order the internal coordinates to generate those produced by
    # smirnoffee.
    ic_by_type = defaultdict(list)

    ic_type_to_name = {
        Distance: "distances",
        Angle: "angles",
        Dihedral: "dihedrals",
        OutOfPlane: "out-of-plane-angles",
    }

    for internal_coordinate in geometric_coordinates.Internals:

        ic_by_type[ic_type_to_name[internal_coordinate.__class__]].append(
            internal_coordinate
        )

    ordered_internals = []

    for ic_type in internal_coordinates_indices:

        ic_by_index = {
            _geometric_internal_coordinate_to_indices(ic): ic
            for ic in ic_by_type[ic_type]
        }

        for ic_indices in internal_coordinates_indices[ic_type]:

            ic_indices = tuple(int(i) for i in ic_indices)

            if ic_indices[-1] > ic_indices[0]:
                ic_indices = tuple(reversed(ic_indices))

            ordered_internals.append(ic_by_index[ic_indices])

    geometric_coordinates.Internals = ordered_internals

    geometric_coordinates.derivatives(conformer.detach().numpy())
    return geometric_coordinates.second_derivatives(conformer.detach().numpy())


def geometric_project_derivatives(
    molecule: Molecule,
    conformer: torch.Tensor,
    internal_coordinates_indices: Dict[str, torch.Tensor],
    reference_gradients: torch.Tensor,
    reference_hessians: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A helper method to project a set of gradients and hessians into internal
    coordinates using ``geomTRIC``.

    Args:
        molecule: The molecule of interest
        conformer: The conformer of the molecule with units of [A] and shape=(n_atoms, 3)
        internal_coordinates_indices: The indices of the atoms involved in each type
            of internal coordinate.
        reference_gradients: The gradients to project.
        reference_hessians: The hessians to project.

    Returns:
        The projected gradients and hessians.
    """

    from geometric.internal import Angle, Dihedral, Distance, OutOfPlane
    from geometric.internal import PrimitiveInternalCoordinates as GeometricPRIC
    from geometric.internal import (
        RotationA,
        RotationB,
        RotationC,
        TranslationX,
        TranslationY,
        TranslationZ,
    )
    from geometric.molecule import Molecule as GeometricMolecule

    geometric_molecule = GeometricMolecule()
    geometric_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.element.symbol for atom in molecule.atoms],
        "bonds": [(bond.atom1_index, bond.atom2_index) for bond in molecule.bonds],
        "name": molecule.name,
        "xyzs": [conformer.detach().numpy()],
    }

    geometric_coordinates = GeometricPRIC(geometric_molecule)

    geometric_coordinates.Internals = [
        internal
        for internal in geometric_coordinates.Internals
        if not isinstance(
            internal,
            (TranslationX, TranslationY, TranslationZ, RotationA, RotationB, RotationC),
        )
    ]

    # We need to re-order the internal coordinates to generate those produced by
    # smirnoffee.
    ic_by_type = defaultdict(list)

    ic_type_to_name = {
        Distance: "distances",
        Angle: "angles",
        Dihedral: "dihedrals",
        OutOfPlane: "out-of-plane-angles",
    }

    for internal_coordinate in geometric_coordinates.Internals:

        ic_by_type[ic_type_to_name[internal_coordinate.__class__]].append(
            internal_coordinate
        )

    ordered_internals = []

    for ic_type in internal_coordinates_indices:

        ic_by_index = {
            _geometric_internal_coordinate_to_indices(ic): ic
            for ic in ic_by_type[ic_type]
        }

        for ic_indices in internal_coordinates_indices[ic_type]:

            ic_indices = tuple(int(i) for i in ic_indices)

            if ic_indices[-1] > ic_indices[0]:
                ic_indices = tuple(reversed(ic_indices))

            ordered_internals.append(ic_by_index[ic_indices])

    geometric_coordinates.Internals = ordered_internals

    reference_gradients = reference_gradients.numpy().flatten()
    reference_hessians = reference_hessians.numpy().reshape(molecule.n_atoms * 3, -1)

    xyz = conformer.detach().numpy()

    return (
        geometric_coordinates.calcGrad(xyz, reference_gradients),
        geometric_coordinates.calcHess(xyz, reference_gradients, reference_hessians),
    )
