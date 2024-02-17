"""Helpers for training parameters."""

import typing

import pydantic
import smee
import torch


def _unflatten_tensors(
    flat_tensor: torch.Tensor, shapes: list[torch.Size]
) -> list[torch.Tensor]:
    """Unflatten a flat tensor into a list of tensors with the given shapes."""

    tensors = []
    start_idx = 0

    for shape in shapes:
        tensors.append(
            flat_tensor[start_idx : start_idx + shape.numel()].reshape(shape)
        )
        start_idx += shape.numel()

    return tensors


class ParameterConfig(pydantic.BaseModel):
    """Configuration for how a potential's parameters should be trained."""

    cols: list[str] = pydantic.Field(
        description="The parameters to train, e.g. 'k', 'length', 'epsilon'."
    )

    scales: dict[str, float] = pydantic.Field(
        {},
        description="The scales to apply to each parameter, e.g. 'k': 1.0, "
        "'length': 1.0, 'epsilon': 1.0.",
    )
    constraints: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        {},
        description="The min and max values to clamp each parameter within, e.g. "
        "'k': (0.0, None), 'angle': (0.0, pi), 'epsilon': (0.0, None), where "
        "none indicates no constraint.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_keys(self):
        """Ensure that the keys in `scales` and `constraints` match `cols`."""

        if any(key not in self.cols for key in self.scales):
            raise ValueError("cannot scale non-trainable parameters")

        if any(key not in self.cols for key in self.constraints):
            raise ValueError("cannot constrain non-trainable parameters")


class Trainable:
    """A wrapper around a SMEE force field that handles zeroing out gradients of
    fixed parameters and applying parameter constraints."""

    @classmethod
    def _prepare(
        cls,
        force_field: smee.TensorForceField,
        config: dict[str, ParameterConfig],
        attr: typing.Literal["parameters", "attributes"],
    ):
        potential_types = sorted(config)
        potentials = [
            force_field.potentials_by_type[potential_type]
            for potential_type in potential_types
        ]

        values = []
        shapes = []

        unfrozen_cols = []
        unfrozen_col_offset = 0

        scales = []

        clamp_lower = []
        clamp_upper = []

        for potential_type, potential in zip(potential_types, potentials):
            potential_values = getattr(potential, attr).detach().clone()
            potential_values_flat = potential_values.flatten()

            shapes.append(potential_values.shape)
            values.append(potential_values_flat)

            potential_cols = getattr(potential, f"{attr[:-1]}_cols")

            n_rows = 1 if attr == "attributes" else len(potential_values)

            unfrozen_cols.extend(
                unfrozen_col_offset + col_idx + row_idx * potential_values.shape[-1]
                for row_idx in range(n_rows)
                for col_idx, col in enumerate(potential_cols)
                if col in config[potential_type].cols
            )

            unfrozen_col_offset += len(potential_values_flat)

            potential_scales = [
                config[potential_type].scales.get(col, 1.0) for col in potential_cols
            ] * n_rows

            scales.extend(potential_scales)

            potential_clamp_lower = [
                config[potential_type].constraints.get(col, (None, None))[0]
                for col in potential_cols
            ] * n_rows
            potential_clamp_lower = [
                -torch.inf if x is None else x for x in potential_clamp_lower
            ]
            clamp_lower.extend(potential_clamp_lower)

            potential_clamp_upper = [
                config[potential_type].constraints.get(col, (None, None))[1]
                for col in potential_cols
            ] * n_rows
            potential_clamp_upper = [
                torch.inf if x is None else x for x in potential_clamp_upper
            ]
            clamp_upper.extend(potential_clamp_upper)

        values = (
            smee.utils.tensor_like([], force_field.potentials[0].parameters)
            if len(values) == 0
            else torch.cat(values)
        )

        return (
            potential_types,
            values,
            shapes,
            torch.tensor(unfrozen_cols),
            smee.utils.tensor_like(scales, values),
            smee.utils.tensor_like(clamp_lower, values),
            smee.utils.tensor_like(clamp_upper, values),
        )

    def __init__(
        self,
        force_field: smee.TensorForceField,
        parameters: dict[str, ParameterConfig],
        attributes: dict[str, ParameterConfig],
    ):
        """

        Args:
            force_field: The force field to wrap.
            parameters: Configure which parameters to train. This should be a dictionary
                with keys of potential types, and values of `ParameterConfig` instances.
            attributes: Configure which attributes to train. This should be a dictionary
                with keys of potential types, and values of `ParameterConfig` instances.
        """
        self._force_field = force_field

        (
            self._param_types,
            param_values,
            self._param_shapes,
            param_unfrozen_idxs,
            param_scales,
            param_clamp_lower,
            param_clamp_upper,
        ) = self._prepare(force_field, parameters, "parameters")
        (
            self._attr_types,
            attr_values,
            self._attr_shapes,
            attr_unfrozen_idxs,
            attr_scales,
            attr_clamp_lower,
            attr_clamp_upper,
        ) = self._prepare(force_field, attributes, "attributes")

        self._values = torch.cat([param_values, attr_values])

        self._unfrozen_idxs = torch.cat(
            [param_unfrozen_idxs, attr_unfrozen_idxs + len(param_scales)]
        ).long()

        self._scales = torch.cat([param_scales, attr_scales])[self._unfrozen_idxs]

        self._clamp_lower = torch.cat([param_clamp_lower, attr_clamp_lower])[
            self._unfrozen_idxs
        ]
        self._clamp_upper = torch.cat([param_clamp_upper, attr_clamp_upper])[
            self._unfrozen_idxs
        ]

    @torch.no_grad()
    def to_values(self) -> torch.Tensor:
        """Returns unfrozen parameter and attribute values as a flat tensor."""
        values_flat = self.clamp(self._values[self._unfrozen_idxs] * self._scales)
        return values_flat.detach().clone().requires_grad_()

    def to_force_field(self, values_flat: torch.Tensor) -> smee.TensorForceField:
        """Returns a force field with the parameters and attributes set to the given
        values.

        Args:
            values_flat: A flat tensor of parameter and attribute values. See
                ``to_values`` for the expected shape and ordering.
        """
        potentials = self._force_field.potentials_by_type

        values = self._values.detach().clone()
        values[self._unfrozen_idxs] = values_flat / self._scales
        values = _unflatten_tensors(values, self._param_shapes + self._attr_shapes)

        params = values[: len(self._param_shapes)]

        for potential_type, param in zip(self._param_types, params, strict=True):
            potentials[potential_type].parameters = param

        attrs = values[len(self._param_shapes) :]

        for potential_type, attr in zip(self._attr_types, attrs, strict=True):
            potentials[potential_type].attributes = attr

        return self._force_field

    @torch.no_grad()
    def clamp(self, values_flat: torch.Tensor) -> torch.Tensor:
        """Clamps the given values to the configured min and max values."""
        return (values_flat / self._scales).clamp(
            min=self._clamp_lower, max=self._clamp_upper
        ) * self._scales
