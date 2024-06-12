"""Utilities for defining loss functions."""

import functools
import logging
import typing

import torch

ClosureFn = typing.Callable[
    [torch.Tensor, bool, bool],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
]
P = typing.ParamSpec("P")

_LOGGER = logging.getLogger(__name__)


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


def combine_closures(
    closures: dict[str, ClosureFn],
    weights: dict[str, float] | None = None,
    verbose: bool = False,
) -> ClosureFn:
    """Combine multiple closures into a single closure.

    Args:
        closures: A dictionary of closure functions.
        weights: Optional dictionary of weights for each closure function.
        verbose: Whether to log the loss of each closure function.

    Returns:
        A combined closure function.
    """

    weights = weights if weights is not None else {name: 1.0 for name in closures}

    if len(closures) == 0:
        raise NotImplementedError("At least one closure function is required.")

    if {*closures} != {*weights}:
        raise ValueError("The closures and weights must have the same keys.")

    def combined_closure_fn(
        x: torch.Tensor, compute_gradient: bool, compute_hessian: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        loss = []
        grad = None if not compute_gradient else []
        hess = None if not compute_hessian else []

        verbose_rows = []

        for name, closure_fn in closures.items():
            local_loss, local_grad, local_hess = closure_fn(
                x, compute_gradient, compute_hessian
            )

            loss.append(weights[name] * local_loss)

            if compute_gradient:
                grad.append(weights[name] * local_grad)
            if compute_hessian:
                hess.append(weights[name] * local_hess)

            if verbose:
                verbose_rows.append(
                    {"target": name, "loss": float(f"{local_loss.item():.5f}")}
                )

        loss = sum(loss[1:], loss[0])

        if compute_gradient:
            grad = sum(grad[1:], grad[0]).detach()
        if compute_hessian:
            hess = sum(hess[1:], hess[0]).detach()

        if verbose:
            import pandas

            _LOGGER.info(
                "loss breakdown:\n"
                + pandas.DataFrame(verbose_rows).to_string(index=False)
            )

        return loss.detach(), grad, hess

    return combined_closure_fn


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
