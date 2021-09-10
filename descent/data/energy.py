import functools
from collections import defaultdict
from multiprocessing import Pool
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from openff.interchange.components.interchange import Interchange
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from smirnoffee.geometry.internal import (
    cartesian_to_internal,
    detect_internal_coordinates,
)
from smirnoffee.potentials.potentials import evaluate_vectorized_system_energy
from torch._vmap_internals import vmap
from torch.autograd import grad
from tqdm import tqdm
from typing_extensions import Literal

from descent import metrics, transforms
from descent.data import Dataset, DatasetEntry
from descent.metrics import LossMetric
from descent.models import ParameterizationModel
from descent.transforms import LossTransform

if TYPE_CHECKING:

    from openff.qcsubmit.results import OptimizationResultCollection
    from qcportal.models import ObjectId

_HARTREE_TO_KJ_MOL = (
    (1.0 * unit.hartree * unit.avogadro_constant)
    .to(unit.kilojoule / unit.mole)
    .magnitude
)
_INVERSE_BOHR_TO_ANGSTROM = (1.0 * unit.bohr ** -1).to(unit.angstrom ** -1).magnitude


class EnergyEntry(DatasetEntry):
    """A object that stores reference energy, gradient and hessian labels for a molecule
    in multiple conforms."""

    def __init__(
        self,
        model_input: Union[Interchange],
        conformers: torch.Tensor,
        reference_energies: Optional[torch.Tensor] = None,
        reference_gradients: Optional[torch.Tensor] = None,
        gradient_coordinate_system: Literal["cartesian", "ric"] = "cartesian",
        reference_hessians: Optional[torch.Tensor] = None,
        hessian_coordinate_system: Literal["cartesian", "ric"] = "cartesian",
    ):
        """

        Args:
            reference_energies: The reference energies with shape=(n_conformers, 1)
                and units of [kJ / mol].
            reference_gradients: The reference gradients with
                shape=(n_conformers, n_atoms, 3) and units of [kJ / mol / A].
            gradient_coordinate_system: The coordinate system to project the QM and MM
                gradients to before computing the loss metric.
            reference_hessians: The reference gradients with
                shape=(n_conformers, n_atoms * 3, n_atoms * 3) and units of
                [kJ / mol / A^2].
            hessian_coordinate_system: The coordinate system to project the QM and MM
                hessians to before computing the loss metric.
        """

        super(EnergyEntry, self).__init__(model_input)

        self._validate_inputs(
            conformers,
            reference_energies,
            reference_gradients,
            reference_hessians,
            model_input,
        )

        self._conformers = conformers

        internal_coordinate_systems = {
            key for key in [gradient_coordinate_system, hessian_coordinate_system]
        }
        self._inverse_b_matrices = {
            coordinate_system.lower(): self._initialize_internal_coordinates(
                coordinate_system, model_input.topology, reference_hessians is not None
            )
            for coordinate_system in internal_coordinate_systems
            if coordinate_system is not None
            and coordinate_system.lower() != "cartesian"
        }

        self._reference_energies = reference_energies

        if reference_hessians is not None:

            reference_hessians = self._project_hessians(
                reference_hessians, reference_gradients, hessian_coordinate_system
            )

        self._reference_hessians = reference_hessians
        self._hessian_coordinate_system = hessian_coordinate_system

        if reference_gradients is not None:

            reference_gradients = self._project_gradients(
                reference_gradients, gradient_coordinate_system
            )

        self._reference_gradients = reference_gradients
        self._gradient_coordinate_system = gradient_coordinate_system

    @classmethod
    def _validate_inputs(
        cls,
        conformers: torch.Tensor,
        reference_energies: Optional[torch.Tensor],
        reference_gradients: Optional[torch.Tensor],
        reference_hessians: Optional[torch.Tensor],
        system: Interchange,
    ):
        """Validate the shapes of the input tensors."""

        if system.topology.n_topology_molecules != 1:
            raise NotImplementedError("only single molecules are supported")

        assert (
            len(conformers.shape) == 3
        ), "conformers must have shape=(n_conformers, n_atoms, 3)"

        n_conformers, n_atoms, _ = conformers.shape

        assert system.topology.n_topology_atoms == n_atoms, (
            "the number of atoms in the interchange must match the number in the "
            "conformer"
        )

        if reference_energies is not None:
            assert (
                n_conformers > 1
            ), "at least two conformers must be provided when training to energies"

        reference_tensors = (
            ([] if reference_energies is None else [reference_energies])
            + ([] if reference_gradients is None else [reference_gradients])
            + ([] if reference_hessians is None else [reference_hessians])
        )
        assert (
            len(reference_tensors) > 0
        ), "at least one type of reference data must be provided"

        assert all(
            len(reference_tensor) == n_conformers
            for reference_tensor in reference_tensors
        ), (
            "the number of conformers and reference energies / "
            "gradients / hessians must match"
        )

        assert reference_energies is None or reference_energies.shape == (
            n_conformers,
            1,
        ), "reference energy tensor must have shape=(n_conformers, 1)"

        assert reference_gradients is None or reference_gradients.shape == (
            n_conformers,
            n_atoms,
            3,
        ), "reference gradient tensor must have shape=(n_conformers, n_atoms, 3)"

        assert reference_hessians is None or reference_hessians.shape == (
            n_conformers,
            n_atoms * 3,
            n_atoms * 3,
        ), (
            "reference hessian tensor must have shape=(n_conformers, n_atoms * 3, "
            "n_atoms * 3)"
        )

    def _initialize_internal_coordinates(
        self,
        coordinate_system: Literal["ric"],
        topology: Topology,
        compute_hessians: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the B, inverse G and B' matrices [1] used to project a set of
        cartesian conformers, gradients and hessians into a particular coordinate system.

        Args:
            coordinate_system:
            topology:
            compute_hessians:

        References:
            1. P. Pulay, G. Fogarasi, F. Pang, and J. E. Boggs J. Am. Chem. Soc. 1979,
               101, 10, 2550–2560

        Returns:
            The B, inverse G and B' matrices with shapes of:

            * (n_conformers, n_internal_coords, n_atoms * 3)
            * (n_conformers, n_internal_coords, n_internal_coords)
            * (n_conformers, n_internal_coords, n_atoms * 3, n_atoms * 3)

            respectively.
        """

        bond_tensor = torch.tensor(
            [
                (bond.atom1_index, bond.atom2_index)
                for bond in next(iter(topology.reference_molecules)).bonds
            ]
        )

        b_matrices = []
        g_inverses = []

        b_matrix_gradients = []

        n_internal_degrees = 0

        for conformer in self._conformers:

            conformer = conformer.detach().clone().requires_grad_()

            internal_coordinate_indices = detect_internal_coordinates(
                conformer, bond_tensor, coordinate_system="ric"
            )

            internal_coordinates = torch.cat(
                [
                    ic_values[1].flatten()
                    for ic_values in cartesian_to_internal(
                        conformer,
                        ic_indices=internal_coordinate_indices,
                        coordinate_system=coordinate_system,
                    ).values()
                ]
            )
            internal_coordinate_indices = [
                atom_indices
                for ic_type in internal_coordinate_indices
                for atom_indices in internal_coordinate_indices[ic_type]
            ]

            b_matrix = torch.zeros(
                (
                    len(internal_coordinate_indices),
                    conformer.shape[0] * conformer.shape[1],
                )
            )
            b_matrix_gradient = (
                torch.zeros(
                    (
                        len(internal_coordinate_indices),
                        *conformer.shape,
                        *conformer.shape,
                    )
                )
                if compute_hessians
                else torch.tensor([])
            )

            for row_index, (atom_indices, row) in enumerate(
                zip(internal_coordinate_indices, internal_coordinates)
            ):

                (gradient,) = grad(row, conformer, create_graph=True)

                # noinspection PyArgumentList
                b_matrix[row_index] = gradient.flatten()

                if not compute_hessians:
                    continue

                # TODO: computing the hessians in this way is still ~4x5 times slower
                #       than geomeTRIC. Explicit equations should likely be used rather
                #       than auto-diffing.
                gradient_subset = gradient[atom_indices].flatten()
                basis_vectors = torch.eye(len(gradient_subset))

                def get_vjp(v):

                    return torch.autograd.grad(
                        gradient_subset, conformer, v, retain_graph=True
                    )[0]

                row_hessian = vmap(get_vjp)(basis_vectors).reshape(
                    (len(atom_indices), 3, *conformer.shape)
                )

                for i, atom_index in enumerate(atom_indices):
                    b_matrix_gradient[row_index, atom_index, :, :, :] = row_hessian[i]

            if len(b_matrix_gradient) > 0:

                b_matrix_gradient = b_matrix_gradient.reshape(
                    b_matrix.shape[0], b_matrix.shape[1], b_matrix.shape[1]
                )

            # rcond was selected here to match geomeTRIC = 0.9.7.2
            g_inverse = torch.pinverse(b_matrix @ b_matrix.T, rcond=1.0e-6)

            b_matrices.append(b_matrix.detach())
            g_inverses.append(g_inverse.detach())

            b_matrix_gradients.append(b_matrix_gradient.detach())

            n_internal_degrees = max(n_internal_degrees, len(g_inverse))

        # We pad the tensors with zeros to ensure that they all have the same
        # dimensions to allow easier batch calculations. This can occur when
        # certain conformers contain different ICs, e.g. one with a planar N and
        # another with a pyramidal N.
        for j, tensor, i, tensors in (
            (j, tensor, i, tensors)
            for i, tensors in enumerate((b_matrices, g_inverses, b_matrix_gradients))
            for j, tensor in enumerate(tensors)
        ):

            if len(tensor) == n_internal_degrees:
                continue

            n_pad = n_internal_degrees - tensor.shape[0]

            tensor = torch.cat((tensor, torch.zeros((n_pad, *tensor.shape[1:]))), dim=0)

            if i == 1:
                tensor = torch.cat(
                    (tensor, torch.zeros((tensor.shape[0], n_pad))), dim=1
                )

            tensors[j] = tensor

        return (
            torch.stack(b_matrices),
            torch.stack(g_inverses),
            torch.stack(b_matrix_gradients),
        )

    def _project_gradients(
        self, gradients: torch.Tensor, coordinate_system: Literal["cartesian", "ric"]
    ) -> torch.Tensor:
        """Projects a set of gradients from cartesian to a specified coordinate
        system."""

        if coordinate_system.lower() == "cartesian":
            return gradients

        b_matrix, g_inverse, _ = self._inverse_b_matrices[coordinate_system.lower()]

        # From doi:10.1002/(sici)1096-987x(19960115)17:1<49::aid-jcc5>3.0.co;2-0
        # Eqn (3) g_q = G^- @ B @ g_x
        gradients = torch.bmm(
            torch.bmm(g_inverse, b_matrix),
            gradients.reshape((len(g_inverse), -1, 1)),
        )

        return gradients

    def _project_hessians(
        self,
        hessians: torch.Tensor,
        gradients: torch.Tensor,
        coordinate_system: Literal["cartesian", "ric"],
    ) -> torch.Tensor:
        """Projects a set of hessians from cartesian to a specified coordinate system."""

        if coordinate_system.lower() == "cartesian":
            return hessians

        assert (
            gradients is not None
        ), "gradients must be provided if using internal coordinate hessians"

        b_matrix, g_inverse, b_matrix_gradient = self._inverse_b_matrices[
            coordinate_system.lower()
        ]

        # See ``_initialize_gradients``.
        gradients = torch.bmm(
            torch.bmm(g_inverse, b_matrix), gradients.reshape((len(g_inverse), -1, 1))
        )
        hessian_delta = (
            hessians
            - torch.bmm(
                b_matrix_gradient.reshape(
                    len(gradients), g_inverse.shape[1], -1
                ).transpose(1, 2),
                gradients,
            ).reshape(hessians.shape)
        )

        hessians = g_inverse

        for matrix in [
            b_matrix,
            hessian_delta,
            b_matrix.transpose(1, 2),
            g_inverse,
        ]:
            hessians = torch.bmm(hessians, matrix)

        return hessians

    def _evaluate_mm_energies(
        self,
        model: ParameterizationModel,
        compute_gradients: bool = False,
        compute_hessians: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the perturbed MM energies, gradients and hessians of the system
        associated with this entry.

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.
        """

        vectorized_system = model.forward(self._model_input)
        conformers = self._conformers.detach().clone().requires_grad_()

        mm_energies, mm_gradients, mm_hessians = [], [], []

        # TODO: replace with either vmap or vectorize smirnoffee
        for conformer in conformers:

            mm_energy = evaluate_vectorized_system_energy(vectorized_system, conformer)
            mm_energies.append(mm_energy)

            if not compute_gradients and not compute_hessians:
                continue

            (mm_gradient,) = torch.autograd.grad(
                mm_energy, conformer, create_graph=compute_gradients or compute_hessians
            )
            mm_gradients.append(mm_gradient)

            if not compute_hessians:
                continue

            # noinspection PyArgumentList
            mm_hessian = torch.cat(
                [
                    torch.autograd.grad(value, conformer, retain_graph=True)[0]
                    for value in mm_gradient.flatten()
                ]
            ).reshape(
                (
                    conformer.shape[0] * conformer.shape[1],
                    conformer.shape[0] * conformer.shape[1],
                )
            )
            mm_hessians.append(mm_hessian)

        return (
            torch.stack(mm_energies),
            None if not compute_gradients else torch.stack(mm_gradients),
            None if not compute_hessians else torch.stack(mm_hessians),
        )

    @staticmethod
    def _evaluate_loss_contribution(
        reference_tensor: torch.Tensor,
        computed_tensor: torch.Tensor,
        data_transforms: Union[LossTransform, List[LossTransform]],
        data_metric: LossMetric,
    ) -> torch.Tensor:
        """Computes the loss contribution for a set of computed and reference labels.

        Args:
            reference_tensor: The reference tensor.
            computed_tensor: The computed tensor.
            data_transforms: Transforms to apply to the reference and computed tensors.
            data_metric: The loss metric (e.g. MSE) to compute.

        Returns:
            The loss contribution.
        """

        transformed_reference_tensor = transforms.transform_tensor(
            reference_tensor, data_transforms
        )
        transformed_computed_tensor = transforms.transform_tensor(
            computed_tensor, data_transforms
        )

        return data_metric(transformed_computed_tensor, transformed_reference_tensor)

    def evaluate(
        self,
        model: ParameterizationModel,
        energy_transforms: Optional[Union[LossTransform, List[LossTransform]]] = None,
        energy_metric: Optional[LossMetric] = None,
        gradient_transforms: Optional[Union[LossTransform, List[LossTransform]]] = None,
        gradient_metric: Optional[LossMetric] = None,
        hessian_transforms: Optional[Union[LossTransform, List[LossTransform]]] = None,
        hessian_metric: Optional[LossMetric] = None,
    ) -> torch.Tensor:
        """

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.
            energy_transforms: Transforms to apply to the QM and MM energies
                before computing the loss metric. By default
                ``descent.transforms.relative(index=0)`` is used if no value is provided.
            energy_metric: The loss metric (e.g. MSE) to compute from the QM and MM
                energies. By default ``descent.metrics.mse()`` is used if no value is
                provided.
            gradient_transforms: Transforms to apply to the QM and MM gradients
                before computing the loss metric. By default
                ``descent.transforms.identity()`` is used if no value is provided.
            gradient_metric: The loss metric (e.g. MSE) to compute from the QM and MM
                gradients. By default ``descent.metrics.mse()`` is used if no value is
                provided.
            hessian_transforms: Transforms to apply to the QM and MM hessians
                before computing the loss metric. By default
                ``descent.transforms.identity()`` is used if no value is provided.
            hessian_metric: The loss metric (e.g. MSE) to compute from the QM and MM
                hessians. By default ``descent.metrics.mse()`` is used if no value is
                provided.
        """
        mm_energies, mm_gradients, mm_hessians = self._evaluate_mm_energies(
            model,
            compute_gradients=(
                self._reference_hessians is not None
                or self._reference_gradients is not None
            ),
            compute_hessians=self._reference_hessians is not None,
        )

        loss = torch.zeros(1)

        if self._reference_energies is not None:

            loss += self._evaluate_loss_contribution(
                self._reference_energies,
                mm_energies,
                energy_transforms
                if energy_transforms is not None
                else transforms.relative(index=0),
                energy_metric if energy_metric is not None else metrics.mse(),
            )

        if self._reference_gradients is not None:

            loss += self._evaluate_loss_contribution(
                self._reference_gradients,
                self._project_gradients(mm_gradients, self._gradient_coordinate_system),
                gradient_transforms
                if gradient_transforms is not None
                else transforms.relative(index=0),
                gradient_metric if gradient_metric is not None else metrics.mse(),
            )

        if self._reference_hessians is not None:

            loss += self._evaluate_loss_contribution(
                self._reference_hessians,
                self._project_hessians(
                    mm_hessians, mm_gradients, self._hessian_coordinate_system
                ),
                hessian_transforms
                if hessian_transforms is not None
                else transforms.identity(),
                hessian_metric if hessian_metric is not None else metrics.mse(),
            )

        return loss


class EnergyDataset(Dataset[EnergyEntry]):
    """A data set that stores reference energy, gradient and hessian labels."""

    @classmethod
    def _retrieve_gradient_and_hessians(
        cls,
        optimization_results: "OptimizationResultCollection",
        include_gradients: bool,
        include_hessians: bool,
        verbose: bool = True,
    ) -> Tuple[
        Dict[Tuple[str, "ObjectId"], torch.Tensor],
        Dict[Tuple[str, "ObjectId"], torch.Tensor],
    ]:
        """Retrieves the hessians and gradients associated with a set of QC optimization
        result records.

        Args:
            optimization_results: The collection of result records whose matching
                gradients and hessians should be retrieved where available.
            include_gradients: Whether to retrieve gradient values.
            include_hessians: Whether to retrieve hessian values.
            verbose: Whether to log progress to the terminal.

        Returns:
            The values of the gradients and hessians (if requested) stored in
            dictionaries with keys of ``(server_address, molecule_id)``.

            Gradient tensors will have shape=(n_atoms, 3) and units of [kJ / mol / A]
            and hessian shape=(n_atoms * 3, n_atoms * 3) and units of [kJ / mol / A^2].
        """

        if not include_hessians and not include_gradients:
            return {}, {}

        basic_result_collection = optimization_results.to_basic_result_collection(
            driver=(
                ([] if not include_gradients else ["gradient"])
                + ([] if not include_hessians else ["hessian"])
            )
        )

        qc_gradients, qc_hessians = {}, {}

        for qc_record, _ in tqdm(
            basic_result_collection.to_records(),
            desc="Pulling gradient / hessian data",
            disable=not verbose,
        ):

            address = qc_record.client.address

            if qc_record.driver == "gradient" and include_gradients:

                qc_gradients[(address, qc_record.molecule)] = torch.from_numpy(
                    qc_record.return_result
                    * _HARTREE_TO_KJ_MOL
                    * _INVERSE_BOHR_TO_ANGSTROM
                ).type(torch.float32)

            elif qc_record.driver == "hessian" and include_hessians:

                qc_hessians[(address, qc_record.molecule)] = torch.from_numpy(
                    qc_record.return_result
                    * _HARTREE_TO_KJ_MOL
                    * _INVERSE_BOHR_TO_ANGSTROM
                    * _INVERSE_BOHR_TO_ANGSTROM
                ).type(torch.float32)

        return qc_gradients, qc_hessians

    @classmethod
    def _from_grouped_results(
        cls,
        grouped_data: Tuple[
            str,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ],
        force_field: ForceField,
        **kwargs,
    ) -> "EnergyEntry":

        cmiles, conformers, qc_energies, qc_gradients, qc_hessians = grouped_data

        molecule = Molecule.from_mapped_smiles(cmiles, allow_undefined_stereo=True)
        system = Interchange.from_smirnoff(force_field, molecule.to_topology())

        return EnergyEntry(
            system,
            conformers,
            reference_energies=qc_energies,
            reference_gradients=qc_gradients,
            reference_hessians=qc_hessians,
            **kwargs,
        )

    @classmethod
    def from_optimization_results(
        cls,
        optimization_results: "OptimizationResultCollection",
        initial_force_field: ForceField,
        include_energies: bool = True,
        include_gradients: bool = False,
        gradient_coordinate_system: Literal["cartesian", "ric"] = "cartesian",
        include_hessians: bool = False,
        hessian_coordinate_system: Literal["cartesian", "ric"] = "cartesian",
        n_processes: int = 1,
        verbose: bool = True,
    ) -> "EnergyDataset":
        """Creates a dataset of energy entries (one per unique molecule) from the
        **final** structures a set of QC optimization results.

        Args:
            optimization_results: The collection of result records.
            initial_force_field: The force field that will be trained.
            include_energies: Whether to include energies.
            include_gradients: Whether to include gradients.
            gradient_coordinate_system: The coordinate system to project the QM and MM
                gradients to before computing the loss metric.
            include_hessians: Whether to include hessians.
            hessian_coordinate_system: The coordinate system to project the QM and MM
                hessians to before computing the loss metric.
            n_processes: The number of processes to parallelize this function across.
            verbose: Whether to log progress to the terminal.

        Returns:
            A dataset of the energy entries.
        """

        from simtk import unit as simtk_unit

        # Group the results by molecule ignoring stereochemistry
        per_molecule_records = defaultdict(list)

        for qc_record, molecule in tqdm(
            optimization_results.to_records(),
            desc="Pulling main optimisation records",
            disable=not verbose,
        ):

            molecule: Molecule = molecule.canonical_order_atoms()
            conformer = molecule.conformers[0].value_in_unit(simtk_unit.angstrom)

            smiles = molecule.to_smiles(
                isomeric=True, explicit_hydrogens=True, mapped=True
            )

            per_molecule_records[smiles].append((qc_record, conformer))

        qc_gradients, qc_hessians = cls._retrieve_gradient_and_hessians(
            optimization_results, include_gradients, include_hessians, verbose
        )

        result_tensors = []

        for cmiles, qc_records in per_molecule_records.items():

            # noinspection PyTypeChecker
            grouped_data = [
                (
                    torch.from_numpy(conformer).type(torch.float32),
                    torch.tensor([qc_record.get_final_energy() * _HARTREE_TO_KJ_MOL]),
                    # There should always be a gradient associated with the record
                    # and so we choose to raise a key error when the record is missing
                    # rather than skipping the entry.
                    None
                    if not include_gradients
                    else qc_gradients[
                        (qc_record.client.address, qc_record.final_molecule)
                    ],
                    qc_hessians.get(
                        (qc_record.client.address, qc_record.final_molecule), None
                    ),
                )
                for qc_record, conformer in qc_records
            ]

            conformers, qm_energies, qm_gradients, qm_hessians = zip(*grouped_data)

            result_tensors.append(
                (
                    cmiles,
                    torch.stack(conformers),
                    torch.stack(qm_energies) if include_energies else None,
                    torch.stack(qm_gradients) if include_gradients else None,
                    torch.stack(qm_hessians) if include_hessians else None,
                )
            )

        with Pool(n_processes) as pool:

            entries = list(
                tqdm(
                    pool.imap(
                        functools.partial(
                            cls._from_grouped_results,
                            force_field=initial_force_field,
                            gradient_coordinate_system=gradient_coordinate_system
                            if include_gradients
                            else None,
                            hessian_coordinate_system=hessian_coordinate_system
                            if include_hessians
                            else None,
                        ),
                        result_tensors,
                    ),
                    total=len(result_tensors),
                    disable=not verbose,
                    desc="Building energy contribution objects.",
                )
            )

        return cls(entries)

    def evaluate(
        self,
        model: ParameterizationModel,
        energy_transforms: Optional[Union[LossTransform, List[LossTransform]]] = None,
        energy_metric: Optional[LossMetric] = None,
        gradient_transforms: Optional[Union[LossTransform, List[LossTransform]]] = None,
        gradient_metric: Optional[LossMetric] = None,
        hessian_transforms: Optional[Union[LossTransform, List[LossTransform]]] = None,
        hessian_metric: Optional[LossMetric] = None,
    ) -> torch.Tensor:
        """Evaluates the contribution to the total loss function of the data stored
        in this set using a specified model.

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.
            energy_transforms: Transforms to apply to the QM and MM energies
                before computing the loss metric. By default
                ``descent.transforms.relative(index=0)`` is used if no value is provided.
            energy_metric: The loss metric (e.g. MSE) to compute from the QM and MM
                energies. By default ``descent.metrics.mse()`` is used if no value is
                provided.
            gradient_transforms: Transforms to apply to the QM and MM gradients
                before computing the loss metric. By default
                ``descent.transforms.identity()`` is used if no value is provided.
            gradient_metric: The loss metric (e.g. MSE) to compute from the QM and MM
                gradients. By default ``descent.metrics.mse()`` is used if no value is
                provided.
            hessian_transforms: Transforms to apply to the QM and MM hessians
                before computing the loss metric. By default
                ``descent.transforms.identity()`` is used if no value is provided.
            hessian_metric: The loss metric (e.g. MSE) to compute from the QM and MM
                hessians. By default ``descent.metrics.mse()`` is used if no value is
                provided.

        Returns:
            The loss contribution of this dataset.
        """

        loss = torch.zeros(1)

        for entry in self._entries:

            loss += entry(
                model,
                energy_transforms=energy_transforms,
                energy_metric=energy_metric,
                gradient_transforms=gradient_transforms,
                gradient_metric=gradient_metric,
                hessian_transforms=hessian_transforms,
                hessian_metric=hessian_metric,
            )

        return loss
