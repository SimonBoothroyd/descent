import copy
from typing import Tuple

import numpy
import pytest
import torch
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from smirnoffee.geometry.internal import detect_internal_coordinates

from descent import metrics, transforms
from descent.data.energy import EnergyDataset, EnergyEntry
from descent.models.smirnoff import SMIRNOFFModel
from descent.tests.geometric import geometric_project_derivatives
from descent.tests.mocking.qcdata import mock_optimization_result_collection
from descent.tests.mocking.systems import generate_mock_hcl_system


@pytest.fixture()
def mock_hcl_conformers() -> torch.Tensor:
    """Creates two mock conformers for HCl - one with a bond length of 1 A and another
    with a bond length of 2 A"""

    return torch.tensor(
        [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )


@pytest.fixture()
def mock_hcl_system():
    return generate_mock_hcl_system()


@pytest.fixture()
def mock_hcl_mm_values() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A set of energies, gradients and hessians analytically computed using
    for the ``mock_hcl_conformers`` and ``mock_hcl_system``.
    """

    return (
        torch.tensor([[0.0], [1.0]]),
        torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        ),
        torch.tensor(
            [
                [
                    [+2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
                    [+0.0, 0.0, 0.0, +0.0, 0.0, 0.0],
                    [+0.0, 0.0, 0.0, +0.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, +2.0, 0.0, 0.0],
                    [+0.0, 0.0, 0.0, +0.0, 0.0, 0.0],
                    [+0.0, 0.0, 0.0, +0.0, 0.0, 0.0],
                ],
                [
                    [+2.0, +0.0, +0.0, -2.0, +0.0, +0.0],
                    [+0.0, +1.0, +0.0, +0.0, -1.0, +0.0],
                    [+0.0, +0.0, +1.0, +0.0, +0.0, -1.0],
                    [-2.0, +0.0, +0.0, +2.0, +0.0, +0.0],
                    [+0.0, -1.0, +0.0, +0.0, +1.0, +0.0],
                    [+0.0, +0.0, -1.0, +0.0, +0.0, +1.0],
                ],
            ]
        ),
    )


def test_initialize_internal_coordinates():
    """Test that the internal coordinate matrices can be correctly constructed and
    padding when different conformers of a molecule have different numbers of internal
    coordinates. See ``test_gradient_hessian_projection`` for a more rigorous
    integration test.
    """

    topology = Molecule.from_mapped_smiles("[H:1][C:2]#[C:3][H:4]").to_topology()

    conformers = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )

    entry = EnergyEntry.__new__(EnergyEntry)
    entry._conformers = conformers

    b_matrix, g_inverse, b_matrix_gradient = entry._initialize_internal_coordinates(
        "ric", topology, True
    )

    # The first conformer will have 6 ICs (3 bonds, 2 angles, 1 dihedral), while the
    # second will have 7 as the dihedral is replaced with 2 linear angle terms. We
    # should expect the first conformer then to have some zero paddings added to match
    # the shape of the second conformer matricies.
    assert b_matrix.shape == (2, 7, 12)
    assert torch.allclose(b_matrix[0, 6], torch.tensor(0.0))
    assert not torch.allclose(b_matrix[1, 6], torch.tensor(0.0))

    assert g_inverse.shape == (2, 7, 7)
    assert torch.allclose(g_inverse[0, 6, :], torch.tensor(0.0))
    assert torch.allclose(g_inverse[0, :, 6], torch.tensor(0.0))

    assert not torch.allclose(g_inverse[1, 6, :], torch.tensor(0.0))
    assert not torch.allclose(g_inverse[1, :, 6], torch.tensor(0.0))

    assert b_matrix_gradient.shape == (2, 7, 12, 12)
    assert torch.allclose(b_matrix_gradient[0, 6], torch.tensor(0.0))
    assert not torch.allclose(b_matrix_gradient[1, 6], torch.tensor(0.0))


def test_gradient_hessian_projection(ethanol, ethanol_conformer, ethanol_system):
    """An integration test of projecting a set of gradients and hessians onto internal
    coordinates. The values are compared against the more established ``geomtric``
    package.
    """

    conformer = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]], requires_grad=True)

    x = 0.5 * conformer.square().sum()

    x.backward(retain_graph=True)
    print(x.grad)

    x.backward()
    print(x.grad)

    internal_coordinate_indices = detect_internal_coordinates(
        ethanol_conformer,
        torch.tensor([(bond.atom1_index, bond.atom2_index) for bond in ethanol.bonds]),
    )

    reference_gradients = torch.rand((1, ethanol.n_atoms, 3))
    reference_hessians = torch.rand((1, ethanol.n_atoms * 3, ethanol.n_atoms * 3))

    expected_gradiant, expected_hessian = geometric_project_derivatives(
        ethanol,
        ethanol_conformer,
        internal_coordinate_indices,
        reference_gradients,
        reference_hessians,
    )

    entry = EnergyEntry(
        ethanol_system,
        ethanol_conformer.reshape(1, len(ethanol_conformer), 3),
        reference_gradients=reference_gradients,
        gradient_coordinate_system="ric",
        reference_hessians=reference_hessians,
        hessian_coordinate_system="ric",
    )

    actual_gradiant = entry._reference_gradients.numpy()
    actual_hessian = entry._reference_hessians.numpy()

    assert numpy.allclose(
        actual_gradiant.reshape(expected_gradiant.shape), expected_gradiant, atol=1.0e-3
    )
    assert numpy.allclose(
        actual_hessian.reshape(expected_hessian.shape), expected_hessian, atol=1.0e-3
    )


@pytest.mark.parametrize("compute_gradients", [True, False])
@pytest.mark.parametrize("compute_hessians", [True, False])
def test_evaluate_mm_energies(
    compute_gradients,
    compute_hessians,
    mock_hcl_conformers,
    mock_hcl_system,
    mock_hcl_mm_values,
):

    entry = EnergyEntry(mock_hcl_system, mock_hcl_conformers, torch.zeros((2, 1)))

    mm_energies, mm_gradients, mm_hessians = entry._evaluate_mm_energies(
        SMIRNOFFModel([], None), compute_gradients, compute_hessians
    )

    expected_energies, expected_gradients, expected_hessians = mock_hcl_mm_values

    assert mm_energies.shape == expected_energies.shape
    assert torch.allclose(mm_energies, expected_energies)

    if compute_gradients:
        assert mm_gradients.shape == expected_gradients.shape
        assert torch.allclose(mm_gradients, expected_gradients)
    else:
        assert mm_gradients is None

    if compute_hessians:
        assert mm_hessians.shape == expected_hessians.shape
        assert torch.allclose(mm_hessians, expected_hessians)
    else:
        assert mm_hessians is None


def test_evaluate_energies(mock_hcl_conformers, mock_hcl_system, mock_hcl_mm_values):

    expected_energies, *_ = mock_hcl_mm_values
    expected_scale = torch.rand(1)

    entry = EnergyEntry(
        mock_hcl_system,
        mock_hcl_conformers,
        reference_energies=expected_energies + torch.ones_like(expected_energies),
    )

    loss = entry.evaluate_loss(
        SMIRNOFFModel([], None),
        energy_transforms=lambda x: expected_scale * x,
        energy_metric=metrics.mse(),
    )

    assert loss.shape == (1,)
    assert torch.isclose(loss, expected_scale.square())


def test_evaluate_gradients(mock_hcl_conformers, mock_hcl_system, mock_hcl_mm_values):

    expected_energies, expected_gradients, _ = mock_hcl_mm_values
    expected_scale = torch.rand(1)

    entry = EnergyEntry(
        mock_hcl_system,
        mock_hcl_conformers,
        # Set a reference energy to make sure gradient contributions don't
        # bleed between loss functions
        reference_energies=expected_energies,
        reference_gradients=expected_gradients + torch.ones_like(expected_gradients),
    )

    loss = entry.evaluate_loss(
        SMIRNOFFModel([], None),
        gradient_transforms=lambda x: expected_scale * x,
        gradient_metric=metrics.mse(()),
    )

    assert loss.shape == (1,)
    assert torch.isclose(loss, expected_scale.square())


def test_evaluate_hessians(mock_hcl_conformers, mock_hcl_system, mock_hcl_mm_values):

    expected_energies, expected_gradients, expected_hessians = mock_hcl_mm_values
    expected_scale = torch.rand(1)

    entry = EnergyEntry(
        mock_hcl_system,
        mock_hcl_conformers,
        # Set a reference energy to make sure gradient contributions don't
        # bleed between loss functions
        reference_energies=expected_energies,
        reference_gradients=expected_gradients,
        reference_hessians=expected_hessians + torch.ones_like(expected_hessians),
    )

    loss = entry.evaluate_loss(
        SMIRNOFFModel([], None),
        hessian_transforms=lambda x: expected_scale * x,
        hessian_metric=metrics.mse(()),
    )

    assert loss.shape == (1,)
    assert torch.isclose(loss, expected_scale.square())


def test_evaluate_loss_contribution():

    reference_tensor = torch.tensor([[1.0], [2.0]])
    computed_tensor = torch.tensor([[4.0], [8.0]])

    loss = EnergyEntry._evaluate_loss_contribution(
        reference_tensor, computed_tensor, transforms.relative(), metrics.mse()
    )

    assert torch.isclose(loss, torch.tensor((4.0 - 1.0) ** 2 * 0.5))


def test_from_grouped_results(mock_hcl_conformers, mock_hcl_mm_values):

    mock_energies, mock_gradients, mock_hessians = mock_hcl_mm_values

    created_term = EnergyDataset._from_grouped_results(
        (
            "[Cl:1][Cl:2]",
            mock_hcl_conformers,
            mock_energies,
            mock_gradients,
            mock_hessians,
        ),
        ForceField("openff_unconstrained-1.0.0.offxml"),
    )

    assert created_term._model_input is not None

    assert torch.allclose(created_term._conformers, mock_hcl_conformers)
    assert torch.allclose(created_term._reference_energies, mock_energies)

    assert torch.allclose(created_term._reference_gradients, mock_gradients)
    assert torch.allclose(created_term._reference_hessians, mock_hessians)


@pytest.mark.parametrize("include_energies", [True, False])
@pytest.mark.parametrize("include_gradients", [True, False])
@pytest.mark.parametrize("include_hessians", [True, False])
def test_from_optimization_results(
    monkeypatch, include_energies, include_gradients, include_hessians
):

    from simtk import unit as simtk_unit

    if not include_energies and not include_gradients and not include_hessians:
        pytest.skip("unsupported combination")

    molecules = []

    for smiles in ["C", "CC"]:

        molecule: Molecule = Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)

        for offset in [1.0, 2.0]:
            shifted_molecule = copy.deepcopy(molecule)
            shifted_molecule.conformers[0] += offset * simtk_unit.angstrom

            molecules.append(shifted_molecule)

    optimization_collection = mock_optimization_result_collection(
        molecules, monkeypatch
    )

    energy_dataset = EnergyDataset.from_optimization_results(
        optimization_collection,
        initial_force_field=ForceField(),
        include_energies=include_energies,
        include_gradients=include_gradients,
        gradient_coordinate_system="cartesian",
        include_hessians=include_hessians,
        hessian_coordinate_system="cartesian",
    )

    assert len(energy_dataset) == 2

    for energy_entry, n_atoms in zip(energy_dataset, [5, 8]):

        if not include_energies:
            assert energy_entry._reference_energies is None
        else:

            assert energy_entry._reference_energies is not None
            assert energy_entry._reference_energies.shape == (2, 1)

            assert not torch.allclose(
                energy_entry._reference_energies,
                torch.zeros_like(energy_entry._reference_energies),
            )

        if not include_gradients:
            assert energy_entry._reference_gradients is None
        else:

            assert energy_entry._reference_gradients is not None
            assert energy_entry._reference_gradients.shape == (2, n_atoms, 3)

            assert not torch.allclose(
                energy_entry._reference_gradients,
                torch.zeros_like(energy_entry._reference_gradients),
            )

        if not include_hessians:
            assert energy_entry._reference_hessians is None
        else:

            assert energy_entry._reference_hessians is not None
            assert energy_entry._reference_hessians.shape == (
                2,
                n_atoms * 3,
                n_atoms * 3,
            )

            assert not torch.allclose(
                energy_entry._reference_hessians,
                torch.zeros_like(energy_entry._reference_hessians),
            )
