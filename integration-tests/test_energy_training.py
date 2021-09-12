import pytest
import torch
from openff.units import unit

from descent import metrics, transforms
from descent.data.energy import EnergyEntry
from descent.models.smirnoff import SMIRNOFFModel
from descent.tests.mocking.systems import generate_mock_hcl_system


def train_parameters(
    loss_function,
    loss_kwargs,
    parameter_delta_ids,
    learning_rate: float = 0.01,
    n_epochs: int = 200,
    verbose: bool = True,
) -> torch.Tensor:

    model = SMIRNOFFModel(parameter_delta_ids, None)

    optimizer = torch.optim.Adam([model.parameter_delta], lr=learning_rate)

    for epoch in range(n_epochs):

        loss = loss_function(model, **loss_kwargs)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch}: loss={loss.item()}")

    return model.parameter_delta


@pytest.mark.parametrize(
    "transform", [transforms.identity(), transforms.relative(index=0)]
)
@pytest.mark.parametrize("metric", [metrics.mse()])
def test_energies_only(transform, metric):

    conformers = torch.tensor(
        [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]]]
    )

    # Define the expected energies assuming that k=2.5 and l=2.0
    reference_energies = torch.tensor(
        [[1.25 * (distance - 2.0) ** 2] for distance in [1.0, 2.5]]
    )

    starting_system = generate_mock_hcl_system(
        bond_k=5.0 * unit.kilojoule / unit.mole / unit.angstrom ** 2,
        bond_length=2.0 * unit.angstrom,
    )
    loss_function = EnergyEntry(
        starting_system,
        conformers,
        reference_energies=reference_energies,
    )

    actual_parameter_delta = train_parameters(
        loss_function,
        dict(
            energy_metric=metric,
            energy_transforms=transform,
        ),
        [("Bonds", "[#1:1]-[#17:2]", "k")],
        learning_rate=0.1,
    )
    expected_parameter_delta = torch.tensor([-2.5])

    assert actual_parameter_delta.shape == expected_parameter_delta.shape
    assert torch.allclose(actual_parameter_delta, expected_parameter_delta)

    print(f"EXPECTED=f{expected_parameter_delta}  ACTUAL=f{actual_parameter_delta}")


@pytest.mark.parametrize("transform", [transforms.identity()])
@pytest.mark.parametrize("metric", [metrics.mse()])
@pytest.mark.parametrize("coordinate_system", ["cartesian", "ric"])
def test_forces_only(transform, metric, coordinate_system):

    conformers = torch.tensor(
        [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]]]
    )

    # Define the expected gradients assuming that k=2.5 and l=2.0
    reference_gradients = torch.tensor(
        [
            [[-2.5 * (distance - 2.0), 0.0, 0.0], [2.5 * (distance - 2.0), 0.0, 0.0]]
            for distance in [1.0, 2.5]
        ]
    )

    starting_system = generate_mock_hcl_system(
        bond_k=5.0 * unit.kilojoule / unit.mole / unit.angstrom ** 2,
        bond_length=2.0 * unit.angstrom,
    )
    loss_function = EnergyEntry(
        starting_system,
        conformers,
        reference_gradients=reference_gradients,
        gradient_coordinate_system=coordinate_system,
    )

    actual_parameter_delta = train_parameters(
        loss_function,
        dict(
            gradient_metric=metric,
            gradient_transforms=transform,
        ),
        [("Bonds", "[#1:1]-[#17:2]", "k")],
        learning_rate=0.1,
    )
    expected_parameter_delta = torch.tensor([-2.5])

    assert actual_parameter_delta.shape == expected_parameter_delta.shape
    assert torch.allclose(actual_parameter_delta, expected_parameter_delta)

    print(f"EXPECTED={expected_parameter_delta}  ACTUAL={actual_parameter_delta}")


@pytest.mark.parametrize(
    "energy_transform", [transforms.identity(), transforms.relative(index=0)]
)
@pytest.mark.parametrize("energy_metric", [metrics.mse()])
@pytest.mark.parametrize(
    "gradient_transform", [transforms.identity(), transforms.relative(index=0)]
)
@pytest.mark.parametrize("gradient_metric", [metrics.mse()])
@pytest.mark.parametrize("gradient_coordinate_system", ["cartesian", "ric"])
def test_energies_and_forces(
    energy_transform,
    energy_metric,
    gradient_transform,
    gradient_metric,
    gradient_coordinate_system,
):

    conformers = torch.tensor(
        [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]]]
    )

    # Define the expected gradients assuming that k=2.5 and l=2.0
    reference_energies = torch.tensor(
        [[1.25 * (distance - 2.0) ** 2] for distance in [1.0, 2.5]]
    )
    reference_gradients = torch.tensor(
        [
            [[-2.5 * (distance - 2.0), 0.0, 0.0], [2.5 * (distance - 2.0), 0.0, 0.0]]
            for distance in [1.0, 2.5]
        ]
    )

    starting_system = generate_mock_hcl_system(
        bond_k=5.0 * unit.kilojoule / unit.mole / unit.angstrom ** 2,
        bond_length=2.0 * unit.angstrom,
    )
    loss_function = EnergyEntry(
        starting_system,
        conformers,
        reference_energies=reference_energies,
        reference_gradients=reference_gradients,
        gradient_coordinate_system=gradient_coordinate_system,
    )

    actual_parameter_delta = train_parameters(
        loss_function,
        dict(
            energy_transforms=energy_transform,
            energy_metric=energy_metric,
            gradient_metric=gradient_metric,
            gradient_transforms=gradient_transform,
        ),
        [("Bonds", "[#1:1]-[#17:2]", "k")],
        learning_rate=0.1,
    )
    expected_parameter_delta = torch.tensor([-2.5])

    assert actual_parameter_delta.shape == expected_parameter_delta.shape
    assert torch.allclose(actual_parameter_delta, expected_parameter_delta)

    print(f"EXPECTED={expected_parameter_delta}  ACTUAL={actual_parameter_delta}")
