"""Levenberg-Marquardt optimizer.

Notes:
    This is a reimplementation of the Levenberg-Marquardt optimizer from the fantastic
    ForceBalance [1] package. The original code is licensed under the BSD 3-clause
    license which can be found in the LICENSE_3RD_PARTY file.

References:
    [1]: https://github.com/leeping/forcebalance/blob/b395fd4b/src/optimizer.py
"""
import logging
import math
import typing

import pydantic
import smee.utils
import torch

_LOGGER = logging.getLogger(__name__)


ClosureFn = typing.Callable[
    [torch.Tensor, bool, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]
CorrectFn = typing.Callable[[torch.Tensor], torch.Tensor]


Mode = typing.Literal["adaptive", "hessian-search"]
_ADAPTIVE, _HESSIAN_SEARCH = typing.get_args(Mode)


class LevenbergMarquardtConfig(pydantic.BaseModel):
    """Configuration for the Levenberg-Marquardt optimizer."""

    type: typing.Literal["levenberg-marquardt"] = "levenberg-marquardt"

    mode: Mode = pydantic.Field(
        _ADAPTIVE, description="The mode to run the optimizer in."
    )

    trust_radius: float = pydantic.Field(
        0.2, description="Target trust radius.", gt=0.0
    )
    trust_radius_min: float = pydantic.Field(0.05, description="Minimum trust radius.")

    min_eigenvalue: float = pydantic.Field(
        1.0e-4,
        description="Lower bound on hessian eigenvalue. If the smallest eigenvalue "
        "is smaller than this, a small amount of steepest descent is mixed in prior "
        "to taking a next step to try and correct this.",
    )
    min_damping_factor: float = pydantic.Field(
        1.0, description="Minimum damping factor.", gt=0.0
    )

    adaptive_factor: float = pydantic.Field(
        0.25,
        description="Adaptive trust radius adjustment factor to use when running in "
        "adaptive mode.",
        gt=0.0,
    )
    adaptive_damping: float = pydantic.Field(
        1.0,
        description="Adaptive trust radius adjustment damping to use when running in "
        "adaptive mode.",
        gt=0.0,
    )

    search_tolerance: float = pydantic.Field(
        1.0e-4,
        description="The tolerance used when searching for the optimal damping factor "
        "with hessian diagonal search (i.e. ``mode='hessian-search'``).",
        gt=0.0,
    )
    search_trust_radius_max: float = pydantic.Field(
        1.0e-3,
        description="The maximum trust radius to use when falling back to a second "
        "line search if the loss would increase after the one.",
        gt=0.0,
    )
    search_trust_radius_factor: float = pydantic.Field(
        0.1,
        description="The factor to scale the trust radius by when falling back to a "
        "second line search.",
        gt=0.0,
    )

    error_tolerance: float = pydantic.Field(
        1.0,
        description="Steps that increase the loss more than this amount are rejected.",
    )

    quality_threshold_low: float = pydantic.Field(
        0.25,
        description="The threshold below which the step is considered low quality.",
    )
    quality_threshold_high: float = pydantic.Field(
        0.75,
        description="The threshold above which the step is considered high quality.",
    )

    max_steps: int = pydantic.Field(
        ..., description="The maximum number of full steps to perform.", gt=0
    )


def _invert_svd(matrix: torch.Tensor, threshold: float = 1e-12) -> torch.Tensor:
    """Invert a matrix using SVD.

    Args:
        matrix: The matrix to invert.
        threshold: The threshold below which singular values are considered zero.

    Returns:
        The inverted matrix.
    """
    u, s, vh = torch.linalg.svd(matrix)

    non_zero_idxs = s > threshold

    s_inverse = torch.zeros_like(s)
    s_inverse[non_zero_idxs] = 1.0 / s[non_zero_idxs]

    return vh.T @ torch.diag(s_inverse) @ u.T


def _solver(
    damping_factor: torch.Tensor, gradient: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the Levenberg–Marquardt step.

    Args:
        damping_factor: The damping factor with ``shape=(1,)``.
        gradient: The gradient with ``shape=(n,)``.
        hessian: The Hessian with ``shape=(n, n)``.

    Returns:
        The step with ``shape=(n,)`` and the expected improvement with ``shape=()``.
    """

    hessian_regular = hessian + (damping_factor - 1) ** 2 * torch.eye(
        len(hessian), device=hessian.device, dtype=hessian.dtype
    )
    hessian_inverse = _invert_svd(hessian_regular)

    dx = -(hessian_inverse @ gradient)
    solution = 0.5 * dx @ hessian @ dx + (dx * gradient).sum()

    return dx, solution


def _damping_factor_loss_fn(
    damping_factor: torch.Tensor,
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    trust_radius: float,
) -> torch.Tensor:
    """Computes the squared difference between the target trust radius and the step size
    proposed by the Levenberg–Marquardt solver.

    This is used when finding the optimal damping factor.

    Args:
        damping_factor: The damping factor with ``shape=(1,)``.
        gradient: The gradient with ``shape=(n,)``.
        hessian: The hessian with ``shape=(n, n)``.
        trust_radius: The target trust radius.

    Returns:
        The squared difference.
    """
    dx, _ = _solver(damping_factor, gradient, hessian)
    dx_norm = torch.linalg.norm(dx)

    _LOGGER.debug(
        f"finding trust radius: length {dx_norm:.4e} (target {trust_radius:.4e})"
    )

    return (dx_norm - trust_radius) ** 2


def _step(
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    trust_radius: torch.Tensor,
    config: LevenbergMarquardtConfig,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Compute the next Levenberg–Marquardt step.

    Args:
        gradient: The gradient with ``shape=(n,)``.
        hessian: The hessian with ``shape=(n, n)``.
        trust_radius: The target trust radius.
        config: The optimizer config.

    Notes:
        * the code to 'excise' certain parameters is for now removed until its clear
          it is needed.
        * only trust region is implemented (i.e., only trust0 > 0 is supported)

    Returns:
        The step with ``shape=(n,)``, the expected improvement with ``shape=()``,
        a boolean indicating whether the damping factor was adjusted, and the damping
        factor with ``shape=(1,)``.
    """
    from scipy import optimize

    eigenvalues, _ = torch.linalg.eigh(hessian)
    eigenvalue_smallest = eigenvalues.min()

    if eigenvalue_smallest < config.min_eigenvalue:
        # Mix in SD step if Hessian minimum eigenvalue is negative - experimental.
        adjacency = (
            max(config.min_eigenvalue, 0.01 * abs(eigenvalue_smallest))
            - eigenvalue_smallest
        )

        _LOGGER.info(
            f"hessian has a small or negative eigenvalue ({eigenvalue_smallest:.1e}), "
            f"mixing in some steepest descent ({adjacency:.1e}) to correct this."
        )
        hessian += adjacency * torch.eye(
            hessian.shape[0], device=hessian.device, dtype=hessian.dtype
        )

    damping_factor = torch.tensor(1.0)

    dx, improvement = _solver(damping_factor, gradient, hessian)
    dx_norm = torch.linalg.norm(dx)

    adjust_damping = bool(dx_norm > trust_radius)

    if adjust_damping:
        # LPW tried a few optimizers and found Brent works well, but also that the
        # tolerance is fractional - if the optimized value is zero it takes a lot of
        # meaningless steps.
        damping_factor = optimize.brent(
            _damping_factor_loss_fn,
            (
                gradient.detach().cpu(),
                hessian.detach().cpu(),
                trust_radius.detach().cpu(),
            ),
            brack=(config.min_damping_factor, config.min_damping_factor * 4),
            tol=1.0e-6 if config.mode.lower() == _ADAPTIVE else 1.0e-4,
        )

        dx, improvement = _solver(damping_factor, gradient, hessian)

    return dx, improvement, adjust_damping, damping_factor


def _hessian_diagonal_search(
    x: torch.Tensor,
    closure: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    closure_fn: ClosureFn,
    correct_fn: CorrectFn,
    damping_factor: torch.Tensor,
    trust_radius: torch.Tensor,
    config: LevenbergMarquardtConfig,
) -> tuple[torch.Tensor, float]:
    """

    Args:
        Args:
        x: The current parameters.
        closure: The loss, gradient and hessian evaluated at ``x``.
        closure_fn: The closure function.
        correct_fn: The parameter 'correction' function.
        damping_factor: The current damping factor.
        trust_radius: The current trust radius.
        config: The optimizer config.

    Returns:
        The step with ``shape=(n,)`` and the expected improvement with ``shape=()``.
    """
    from scipy import optimize

    loss, gradient, hessian = closure

    def search_fn(factor: torch.Tensor):
        dx_next, _ = _solver(factor, gradient, hessian)
        x_next = correct_fn(dx_next + x).requires_grad_(x.requires_grad)

        loss_micro, _, _ = closure_fn(x_next, False, False)
        return loss_micro - loss

    damping_factor, expected_improvement, _, _ = optimize.brent(
        search_fn,
        (),
        (damping_factor, damping_factor * 4),
        config.search_tolerance,
        True,
    )

    if expected_improvement > 0.0:
        trust_radius = min(
            config.search_trust_radius_factor * trust_radius,
            config.search_trust_radius_max,
        )
        brent_args = (gradient.detach().cpu(), hessian.detach().cpu(), trust_radius)

        damping_factor = optimize.brent(
            _damping_factor_loss_fn,
            brent_args,
            (config.min_damping_factor, config.min_damping_factor * 4),
            1e-6,
        )

        dx, _ = _solver(damping_factor, gradient, hessian)
        dx_norm = torch.linalg.norm(dx)

        _LOGGER.info(f"restarting search with step size {dx_norm}")

        damping_factor, expected_improvement, _, _ = optimize.brent(
            search_fn,
            (),
            (damping_factor, damping_factor * 4),
            config.search_tolerance,
            True,
        )

    dx, _ = _solver(damping_factor, gradient, hessian)
    return dx, expected_improvement


def _reduce_trust_radius(
    dx_norm: torch.Tensor, config: LevenbergMarquardtConfig
) -> torch.Tensor:
    """Reduce the trust radius.

    Args:
        dx_norm: The size of the previous step.
        config: The optimizer config.

    Returns:
        The reduced trust radius.
    """
    trust_radius = max(
        dx_norm * (1.0 / (1.0 + config.adaptive_factor)), config.trust_radius_min
    )
    _LOGGER.info(f"reducing trust radius to {trust_radius:.4e}")

    return smee.utils.tensor_like(trust_radius, dx_norm)


def _update_trust_radius(
    dx_norm: torch.Tensor,
    step_quality: float,
    trust_radius: torch.Tensor,
    damping_adjusted: bool,
    config: LevenbergMarquardtConfig,
) -> torch.Tensor:
    """Adjust the trust radius based on the quality of the previous step.

    Args:
        dx_norm: The size of the previous step.
        step_quality: The quality of the previous step.
        trust_radius: The current trust radius.
        damping_adjusted: Whether the LM damping factor was adjusted during the
            previous step.
        config: The optimizer config.

    Returns:
        The updated trust radius.
    """

    if step_quality <= config.quality_threshold_low:
        trust_radius = max(
            dx_norm * (1.0 / (1.0 + config.adaptive_factor)),
            smee.utils.tensor_like(config.trust_radius_min, dx_norm),
        )
        _LOGGER.info(
            f"low quality step detected - reducing trust radius to {trust_radius:.4e}"
        )

    elif step_quality >= config.quality_threshold_high and damping_adjusted:
        trust_radius += (
            config.adaptive_factor
            * trust_radius
            * math.exp(
                -config.adaptive_damping * (trust_radius / config.trust_radius - 1.0)
            )
        )
        _LOGGER.info(f"updating trust radius to {trust_radius: .4e}")

    return trust_radius


@torch.no_grad()
def levenberg_marquardt(
    x: torch.Tensor,
    closure_fn: ClosureFn,
    correct_fn: CorrectFn | None = None,
    config: LevenbergMarquardtConfig | None = None,
) -> torch.Tensor:
    """Optimize a given set of parameters using the Levenberg-Marquardt algorithm.

    Notes:
        * This optimizer assumes a least-square loss function.
        * This is a reimplementation of the Levenberg-Marquardt optimizer from the
          ForceBalance package, and so may differ from a standard implementation.

    Args:
        x: The initial guess of the parameters with ``shape=(n,)``.
        closure_fn: A function that computes the loss (``shape=()``), its
            gradient (``shape=(n,)``), and hessian (``shape=(n, n)``)..
        correct_fn: A function that can be used to correct the parameters after
            each step is taken and before the new loss is computed. This may
            include, for example, ensuring that vdW parameters are all positive.
        config: The optimizer config.

    Returns:
        The optimized parameters.
    """

    x = x.clone().detach().requires_grad_(x.requires_grad)

    correct_fn = correct_fn if correct_fn is not None else lambda y: y
    closure_fn = torch.enable_grad()(closure_fn)

    closure_prev = closure_fn(x, True, True)
    trust_radius = torch.tensor(config.trust_radius).to(x.device)

    for step in range(config.max_steps):
        loss_prev, gradient_prev, hessian_prev = closure_prev

        dx, expected_improvement, damping_adjusted, damping_factor = _step(
            gradient_prev, hessian_prev, trust_radius, config
        )

        if config.mode.lower() == _HESSIAN_SEARCH:
            dx, expected_improvement = _hessian_diagonal_search(
                x,
                closure_prev,
                closure_fn,
                correct_fn,
                damping_factor,
                trust_radius,
                config,
            )

        dx_norm = torch.linalg.norm(dx)
        _LOGGER.info(f"{config.mode} step found (length {dx_norm:.4e})")

        x_next = correct_fn(x + dx).requires_grad_(x.requires_grad)

        loss, gradient, hessian = closure_fn(x_next, True, True)
        loss_delta = loss - loss_prev

        step_quality = loss_delta / expected_improvement
        accept_step = True

        if loss > (loss_prev + config.error_tolerance):
            # reject the 'bad' step and try again from where we were
            loss, gradient, hessian = (loss_prev, gradient_prev, hessian_prev)
            trust_radius = _reduce_trust_radius(dx_norm, config)

            accept_step = False
        elif config.mode.lower() == _ADAPTIVE:
            # this was a 'good' step - we can maybe increase the trust radius
            trust_radius = _update_trust_radius(
                dx_norm, step_quality, trust_radius, damping_adjusted, config
            )

        if accept_step:
            x.data.copy_(x_next.data)

        closure_prev = (loss, gradient, hessian)

        _LOGGER.info(f"step={step} loss={loss.detach().cpu().item()}")

    return x
