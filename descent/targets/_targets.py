import torch

import descent.optim


def combine_closures(
    closures: dict[str, descent.optim.ClosureFn],
    weights: dict[str, float] | None = None,
) -> descent.optim.ClosureFn:
    """Combine multiple closures into a single closure.

    Args:
        closures: A dictionary of closure functions.
        weights: Optional dictionary of weights for each closure function.

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

        for name, closure_fn in closures.items():

            local_loss, local_grad, local_hess = closure_fn(
                x, compute_gradient, compute_hessian
            )

            loss.append(weights[name] * local_loss)

            if compute_gradient:
                grad.append(weights[name] * local_grad)
            if compute_hessian:
                hess.append(weights[name] * local_hess)

        loss = sum(loss[1:], loss[0])

        if compute_gradient:
            grad = sum(grad[1:], grad[0])
        if compute_hessian:
            hess = sum(hess[1:], hess[0])

        return loss.detach(), grad, hess

    return combined_closure_fn
