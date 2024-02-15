"""Utilities for defining loss functions."""

import functools
import typing

import torch

ClosureFn = typing.Callable[
    [torch.Tensor, bool, bool],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
]
P = typing.ParamSpec("P")


def to_closure(
    loss_fn: typing.Callable[typing.Concatenate[torch.Tensor, P], torch.Tensor],
    *args: P.args,
    **kwargs: P.kwargs,
) -> ClosureFn:
    """Convert a loss function to a closure function used by second-order optimizers.

    Args:
        loss_fn: The loss function to convert. This should take in a tensor of
            parameters with ``shape=(n,)``, and optionally a set of ``args`` and
            ``kwargs``.
        *args: Positional arguments passed to `loss_fn`.
        **kwargs: Keyword arguments passed to `loss_fn`.

    Returns:
        A closure function that takes in a tensor of parameters with ``shape=(n,)``,
        a boolean flag indicating whether to compute the gradient, and a boolean flag
        indicating whether to compute the Hessian. It returns a tuple of the loss
        value, the gradient, and the Hessian.
    """

    loss_fn_wrapped = functools.partial(loss_fn, *args, **kwargs)

    def closure_fn(
        x: torch.Tensor, compute_gradient: bool, compute_hessian: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        loss = loss_fn_wrapped(x)
        gradient, hessian = None, None

        if compute_hessian:
            hessian = torch.autograd.functional.hessian(
                loss_fn_wrapped, x, vectorize=True, create_graph=False
            ).detach()
        if compute_gradient:
            (gradient,) = torch.autograd.grad(loss, x, create_graph=False)
            gradient = gradient.detach()

        return loss.detach(), gradient, hessian

    return closure_fn


def approximate_hessian(x: torch.Tensor, y_pred: torch.Tensor):
    """Compute the outer product approximation of the hessian of a least squares
    loss function of the sum ``sum((y_pred - y_ref)**2)``.

    Args:
        x: The parameter tensor with ``shape=(n_parameters,)``.
        y_pred: The values predicted using ``x`` with ``shape=(n_predications,)``.

    Returns:
        The outer product approximation of the hessian with ``shape=n_parameters
    """

    y_pred_grad = [torch.autograd.grad(y, x, retain_graph=True)[0] for y in y_pred]
    y_pred_grad = torch.stack(y_pred_grad, dim=0)

    return (
        2.0 * torch.einsum("bi,bj->bij", y_pred_grad, y_pred_grad).sum(dim=0)
    ).detach()
