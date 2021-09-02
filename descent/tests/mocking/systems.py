from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import OFFBioTop
from openff.interchange.components.potentials import Potential
from openff.interchange.components.smirnoff import SMIRNOFFBondHandler
from openff.interchange.models import PotentialKey, TopologyKey
from openff.toolkit.topology import Molecule
from openff.units import unit


def generate_mock_hcl_system(bond_k=None, bond_length=None):
    """Creates an interchange object for HCl that contains a single bond parameter with
    l=1 A and k = 2 kJ / mol by default
    """

    system = Interchange()

    system.topology = OFFBioTop()
    system.topology.copy_initializer(
        Molecule.from_mapped_smiles("[H:1][Cl:2]").to_topology()
    )

    bond_k = (
        bond_k
        if bond_k is not None
        else 2.0 * unit.kilojoule / unit.mole / unit.angstrom ** 2
    )
    bond_length = bond_length if bond_length is not None else 1.0 * unit.angstrom

    system.add_handler(
        "Bonds",
        SMIRNOFFBondHandler(
            slot_map={
                TopologyKey(atom_indices=(0, 1)): PotentialKey(
                    id="[#1:1]-[#17:2]", associated_handler="Bonds"
                )
            },
            potentials={
                PotentialKey(
                    id="[#1:1]-[#17:2]", associated_handler="Bonds"
                ): Potential(parameters={"k": bond_k, "length": bond_length})
            },
        ),
    )

    return system
