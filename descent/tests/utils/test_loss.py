import torch

from descent.utils.loss import approximate_hessian, to_closure


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
