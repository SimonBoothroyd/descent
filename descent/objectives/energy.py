from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from openff.qcsubmit.results import OptimizationResultCollection
from openff.qcsubmit.results.caching import cached_query_procedures
from openff.system.components.system import System
from openff.system.models import PotentialKey
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from smirnoffee.potentials.potentials import evaluate_vectorized_system_energy
from smirnoffee.smirnoff import vectorize_system

from descent.objectives import ObjectiveContribution


class RelativeEnergyObjective(ObjectiveContribution):
    """An objective term which measures the L2 deviations of a set of relative MM
    energies to a set of relative QM energies and, optionally, gradients of such
    energies.
    """

    @property
    def parameter_ids(self) -> List[Tuple[str, PotentialKey, str]]:

        return sorted(
            {
                (handler_type, potential_key, attribute)
                for (handler_type, _), (_, _, parameters) in self._system.items()
                for potential_key, attributes in parameters
                for attribute in attributes
            },
            key=lambda x: x[0],
            reverse=True,
        )

    def __init__(
        self,
        system: System,
        conformers: torch.Tensor,
        qm_energies: torch.Tensor,
        qm_gradients: Optional[torch.Tensor],
    ):

        assert (
            len(conformers.shape) == 3
        ), "conformers must have shape=(n_conformers, n_atoms, 3)"
        n_conformers, n_atoms, _ = conformers.shape

        assert n_conformers > 1, "at least two conformers must be provided"

        assert (
            len(qm_energies) == n_conformers
        ), "number of QM energies must match number of conformers."

        if qm_gradients is not None:

            assert qm_gradients.shape == (
                n_conformers,
                n_atoms,
                3,
            ), "QM gradient tensor must have shape=(n_conformers, n_atoms, 3)"

        self._conformers = conformers

        self._qm_relative_energies = self._absolute_to_relative_energies(
            qm_energies.ravel()
        )
        self._qm_gradients = qm_gradients

        self._system = vectorize_system(system)

    @staticmethod
    def _absolute_to_relative_energies(absolute_energies: torch.Tensor) -> torch.Tensor:
        """Returns a flat tensor of all unique pairs of relative energies"""

        relative_energies = absolute_energies[:, None] - absolute_energies[None, :]

        return relative_energies[
            torch.triu(torch.ones_like(relative_energies), diagonal=1) == 1
        ]

    def evaluate(
        self,
        parameter_delta: torch.Tensor,
        parameter_delta_ids: List[Tuple[str, PotentialKey, str]],
    ) -> torch.Tensor:
        """Evaluate the objective at the given parameter offsets"""

        if self._qm_gradients is not None:
            raise NotImplementedError()

        mm_energies = torch.cat(
            # TODO: Vectorize when SMIRNOFFEE has support.
            [
                evaluate_vectorized_system_energy(
                    self._system, conformer, parameter_delta, parameter_delta_ids
                )
                for conformer in self._conformers
            ]
        )
        mm_relative_energies = self._absolute_to_relative_energies(mm_energies)

        delta = self._qm_relative_energies - mm_relative_energies

        objective = (delta * delta).mean()
        return objective

    @classmethod
    def from_optimization_results(
        cls,
        optimization_results: OptimizationResultCollection,
        initial_force_field: ForceField,
        include_gradients: bool = False,
    ) -> List["RelativeEnergyObjective"]:
        """ """

        from simtk import unit as simtk_unit

        final_record_ids = defaultdict(set)

        # Group the results by molecule ignoring stereochemistry
        per_molecule_records = defaultdict(list)

        for qc_record, molecule in optimization_results.to_records():

            molecule: Molecule = molecule.canonical_order_atoms()
            conformer = molecule.conformers[0].value_in_unit(simtk_unit.angstrom)

            smiles = molecule.to_smiles(isomeric=False, mapped=True)

            per_molecule_records[smiles].append((qc_record, conformer))

            if include_gradients:
                final_record_ids[qc_record.client.address].add(qc_record.trajectory[-1])

        hatree_to_kj_mol = (
            (1.0 * unit.hartree * unit.avogadro_constant)
            .to(unit.kilojoule / unit.mole)
            .magnitude
        )

        qc_gradients = {
            (address, record.id): torch.from_numpy(
                record.return_result * hatree_to_kj_mol
            )
            for address, record_ids in final_record_ids.items()
            for record in cached_query_procedures(address, [*record_ids])
        }
        qc_gradients.setdefault(None)

        contributions = []

        for cmiles, qc_records in per_molecule_records.items():

            molecule = Molecule.from_mapped_smiles(cmiles, allow_undefined_stereo=True)

            # Parameterize the molecule.
            system = System.from_smirnoff(initial_force_field, molecule.to_topology())

            conformer_data = [
                (
                    torch.from_numpy(conformer),
                    torch.tensor([qc_record.get_final_energy() * hatree_to_kj_mol]),
                    qc_gradients.get(
                        (qc_record.client.address, qc_record.trajectory[-1]), None
                    ),
                )
                for qc_record, conformer in qc_records
            ]

            conformers, qm_energies, qm_gradients = zip(*conformer_data)

            contributions.append(
                RelativeEnergyObjective(
                    system,
                    torch.stack(conformers),
                    torch.stack(qm_energies),
                    torch.stack(qm_gradients) if include_gradients else None,
                )
            )

        return contributions
