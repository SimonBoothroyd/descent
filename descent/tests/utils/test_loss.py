import pytest
import torch

from descent.utils.loss import approximate_hessian, combine_closures, to_closure


def test_to_closure():
    def mock_loss_fn(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
        return (a * x**2 + b).sum()

    closure_fn = to_closure(mock_loss_fn, a=2.0, b=3.0)

    theta = torch.Tensor([1.0, 2.0]).requires_grad_(True)

    expected_loss = torch.tensor(16.0)
    expected_grad = torch.tensor([4.0, 8.0])
    expected_hess = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

    loss, grad, hess = closure_fn(theta, True, True)

    assert loss.shape == expected_loss.shape
    assert torch.allclose(loss, expected_loss)

    assert grad.shape == expected_grad.shape
    assert torch.allclose(grad, expected_grad)

    assert hess.shape == expected_hess.shape
    assert torch.allclose(hess, expected_hess)

    loss, grad, hess = closure_fn(theta, False, True)
    assert loss is not None
    assert grad is None
    assert hess is not None

    loss, grad, hess = closure_fn(theta, True, False)
    assert loss is not None
    assert grad is not None
    assert hess is None


def test_combine_closures():

    def mock_closure_a(x_, compute_gradient, compute_hessian):
        loss = x_[0] ** 2
        grad = 2 * x_[0] if compute_gradient else None
        hess = 2.0 if compute_hessian else None

        return loss, grad, hess

    def mock_closure_b(x_, compute_gradient, compute_hessian):
        loss = x_[0] ** 3
        grad = 3 * x_[0] ** 2 if compute_gradient else None
        hess = 6 * x_[0] if compute_hessian else None

        return loss, grad, hess

    closures = {"a": mock_closure_a, "b": mock_closure_b}

    weights = {"a": 1.0, "b": 2.0}

    combined_closure_fn = combine_closures(closures, weights, verbose=True)

    x = torch.tensor([2.0], requires_grad=True)

    loss, grad, hess = combined_closure_fn(
        x, compute_gradient=True, compute_hessian=True
    )

    assert loss == pytest.approx(2.0**2 + 2.0 * 2.0**3)
    assert grad == pytest.approx(2 * 2.0 + 2.0 * 3 * 2.0**2)
    assert hess == pytest.approx(2.0 + 2.0 * 6 * 2.0)


def test_approximate_hessian():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y_pred = 5.0 * x**2 + 3.0 * x + 2.0

    actual_hessian = approximate_hessian(x, y_pred)
    expected_hess = torch.tensor(
        [
            [338.0, 0.0, 0.0, 0.0],
            [0.0, 1058.0, 0.0, 0.0],
            [0.0, 0.0, 2178.0, 0.0],
            [0.0, 0.0, 0.0, 3698.0],
        ]
    )

    assert actual_hessian.shape == expected_hess.shape
    assert torch.allclose(actual_hessian, expected_hess)
