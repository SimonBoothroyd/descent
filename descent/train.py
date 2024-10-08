"""Helpers for training parameters."""

import typing

import openff.interchange.models
import pydantic
import smee
import smee.utils
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


if pydantic.__version__.startswith("1."):
    _PotentialKey = openff.interchange.models.PotentialKey
    PotentialKeyList = list[_PotentialKey]
else:

    class _PotentialKey(pydantic.BaseModel):
        """

        TODO: Needed until interchange upgrades to pydantic >=2
        """

        id: str
        mult: int | None = None
        associated_handler: str | None = None
        bond_order: float | None = None

        def __hash__(self) -> int:
            return hash((self.id, self.mult, self.associated_handler, self.bond_order))

        def __eq__(self, other: object) -> bool:
            import openff.interchange.models

            return (
                isinstance(
                    other, (_PotentialKey, openff.interchange.models.PotentialKey)
                )
                and self.id == other.id
                and self.mult == other.mult
                and self.associated_handler == other.associated_handler
                and self.bond_order == other.bond_order
            )

    def _convert_keys(value: typing.Any) -> typing.Any:
        if not isinstance(value, list):
            return value

        value = [
            _PotentialKey(**v.dict())
            if isinstance(v, openff.interchange.models.PotentialKey)
            else v
            for v in value
        ]
        return value

    PotentialKeyList = typing.Annotated[
        list[_PotentialKey], pydantic.BeforeValidator(_convert_keys)
    ]


class AttributeConfig(pydantic.BaseModel):
    """Configuration for how a potential's attributes should be trained."""

    cols: list[str] = pydantic.Field(
        description="The parameters to train, e.g. 'k', 'length', 'epsilon'."
    )

    scales: dict[str, float] = pydantic.Field(
        {},
        description="The scales to apply to each parameter, e.g. 'k': 1.0, "
        "'length': 1.0, 'epsilon': 1.0.",
    )
    limits: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        {},
        description="The min and max values to clamp each parameter within, e.g. "
        "'k': (0.0, None), 'angle': (0.0, pi), 'epsilon': (0.0, None), where "
        "none indicates no constraint.",
    )

    if pydantic.__version__.startswith("1."):

        @pydantic.root_validator
        def _validate_keys(cls, values):
            cols = values.get("cols")

            scales = values.get("scales")
            limits = values.get("limits")

            if any(key not in cols for key in scales):
                raise ValueError("cannot scale non-trainable parameters")
            if any(key not in cols for key in limits):
                raise ValueError("cannot clamp non-trainable parameters")

            return values

    else:

        @pydantic.model_validator(mode="after")
        def _validate_keys(self):
            """Ensure that the keys in `scales` and `limits` match `cols`."""

            if any(key not in self.cols for key in self.scales):
                raise ValueError("cannot scale non-trainable parameters")

            if any(key not in self.cols for key in self.limits):
                raise ValueError("cannot clamp non-trainable parameters")

            return self


class ParameterConfig(AttributeConfig):
    """Configuration for how a potential's parameters should be trained."""

    include: PotentialKeyList | None = pydantic.Field(
        None,
        description="The keys (see ``smee.TensorPotential.parameter_keys`` for "
        "details) corresponding to specific parameters to be trained. If ``None``, "
        "all parameters will be trained.",
    )
    exclude: PotentialKeyList | None = pydantic.Field(
        None,
        description="The keys (see ``smee.TensorPotential.parameter_keys`` for "
        "details) corresponding to specific parameters to be excluded from training. "
        "If ``None``, no parameters will be excluded.",
    )

    if pydantic.__version__.startswith("1."):

        @pydantic.root_validator
        def _validate_include_exclude(cls, values):
            include = values.get("include")
            exclude = values.get("exclude")

            if include is not None and exclude is not None:
                include = {*include}
                exclude = {*exclude}

                if include & exclude:
                    raise ValueError("cannot include and exclude the same parameter")

            return values

    else:

        @pydantic.model_validator(mode="after")
        def _validate_include_exclude(self):
            """Ensure that the keys in `include` and `exclude` are disjoint."""

            if self.include is not None and self.exclude is not None:
                include = {*self.include}
                exclude = {*self.exclude}

                if include & exclude:
                    raise ValueError("cannot include and exclude the same parameter")

            return self


class Trainable:
    """A convenient wrapper around a tensor force field that gives greater control
    over how parameters should be trained.

    This includes imposing limits on the values of parameters, scaling the values
    so parameters passed to the optimizer have similar magnitudes, and freezing
    parameters so they are not updated during training.
    """

    @classmethod
    def _prepare(
        cls,
        force_field: smee.TensorForceField,
        config: dict[str, AttributeConfig],
        attr: typing.Literal["parameters", "attributes"],
    ):
        """Prepare the trainable parameters or attributes for the given force field and
        configuration."""
        potential_types = sorted(config)
        potentials = [
            force_field.potentials_by_type[potential_type]
            for potential_type in potential_types
        ]

        values = []
        shapes = []

        unfrozen_idxs = []
        unfrozen_col_offset = 0

        scales = []

        clamp_lower = []
        clamp_upper = []

        for potential_type, potential in zip(potential_types, potentials, strict=True):
            potential_config = config[potential_type]

            potential_cols = getattr(potential, f"{attr[:-1]}_cols")
            assert (
                len({*potential_config.cols} - {*potential_cols}) == 0
            ), f"unknown columns: {potential_cols}"

            potential_values = getattr(potential, attr).detach().clone()
            potential_values_flat = potential_values.flatten()

            shapes.append(potential_values.shape)
            values.append(potential_values_flat)

            n_rows = 1 if attr == "attributes" else len(potential_values)

            unfrozen_rows = set(range(n_rows))

            if isinstance(potential_config, ParameterConfig):
                all_keys = [
                    _PotentialKey(**v.dict())
                    for v in getattr(potential, f"{attr[:-1]}_keys")
                ]

                excluded_keys = potential_config.exclude or []
                unfrozen_keys = potential_config.include or all_keys

                key_to_row = {key: row_idx for row_idx, key in enumerate(all_keys)}
                assert len(key_to_row) == len(all_keys), "duplicate keys found"

                unfrozen_rows = {
                    key_to_row[key] for key in unfrozen_keys if key not in excluded_keys
                }

            unfrozen_idxs.extend(
                unfrozen_col_offset + col_idx + row_idx * potential_values.shape[-1]
                for row_idx in range(n_rows)
                if row_idx in unfrozen_rows
                for col_idx, col in enumerate(potential_cols)
                if col in potential_config.cols
            )

            unfrozen_col_offset += len(potential_values_flat)

            potential_scales = [
                potential_config.scales.get(col, 1.0) for col in potential_cols
            ] * n_rows

            scales.extend(potential_scales)

            potential_clamp_lower = [
                potential_config.limits.get(col, (None, None))[0]
                for col in potential_cols
            ] * n_rows
            potential_clamp_lower = [
                -torch.inf if x is None else x for x in potential_clamp_lower
            ]
            clamp_lower.extend(potential_clamp_lower)

            potential_clamp_upper = [
                potential_config.limits.get(col, (None, None))[1]
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
            torch.tensor(unfrozen_idxs),
            smee.utils.tensor_like(scales, values),
            smee.utils.tensor_like(clamp_lower, values),
            smee.utils.tensor_like(clamp_upper, values),
        )

    def _prepare_vsites(
        self, force_field: smee.TensorForceField, config: ParameterConfig
    ):
        """
        Prepare the vsite parameters for optimisation.

        Args:
            force_field: The tensor force field with parameters
                which should be optimised.
            config: The config of the parameters to train.

        Returns:

        """
        vsite_parameters = force_field.v_sites.parameters.detach().clone()
        n_rows = vsite_parameters.shape[0]
        vsite_parameters_flat = vsite_parameters.flatten()
        # define the cols as they are not on the tensor model
        vsite_cols = ["distance", "inPlaneAngle", "outOfPlaneAngle"]

        all_keys = [_PotentialKey(**key.dict()) for key in force_field.v_sites.keys]
        excluded_keys = config.exclude or []
        unfrozen_keys = config.include or all_keys

        key_to_row = {key: row_idx for row_idx, key in enumerate(all_keys)}
        assert len(key_to_row) == len(all_keys), "duplicate keys found"

        unfrozen_rows = {
            key_to_row[key] for key in unfrozen_keys if key not in excluded_keys
        }

        unfrozen_idxs = [
            col_idx + row_idx * vsite_parameters.shape[1]
            for row_idx in range(n_rows)
            if row_idx in unfrozen_rows
            # the vsite model has no parameter cols so define here
            for col_idx, col in enumerate(vsite_cols)
            if col in config.cols
        ]
        vsite_scales = [config.scales.get(col, 1.0) for col in vsite_cols] * n_rows
        clamp_lower = [
            config.limits.get(col, (None, None))[0] for col in vsite_cols
        ] * n_rows
        clamp_lower = [-torch.inf if x is None else x for x in clamp_lower]
        clamp_upper = [
            config.limits.get(col, (None, None))[1] for col in vsite_cols
        ] * n_rows
        clamp_upper = [torch.inf if x is None else x for x in clamp_upper]
        return (
            vsite_parameters_flat,
            torch.tensor(unfrozen_idxs),
            smee.utils.tensor_like(vsite_scales, vsite_parameters),
            smee.utils.tensor_like(clamp_lower, vsite_parameters),
            smee.utils.tensor_like(clamp_upper, vsite_parameters),
        )

    def __init__(
        self,
        force_field: smee.TensorForceField,
        parameters: dict[str, ParameterConfig],
        attributes: dict[str, AttributeConfig],
        vsites: ParameterConfig | None = None,
    ):
        """

        Args:
            force_field: The force field to wrap.
            parameters: Configure which parameters to train.
            attributes: Configure which attributes to train.
            vsites: Configure which vsite parameters to train.
        """
        self._force_field = force_field
        self._fit_vsites = False

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

        values = [param_values, attr_values]
        unfrozen_idxs = [param_unfrozen_idxs, attr_unfrozen_idxs + len(param_scales)]
        scales = [param_scales, attr_scales]
        clamp_lower = [param_clamp_lower, attr_clamp_lower]
        clamp_upper = [param_clamp_upper, attr_clamp_upper]

        if vsites is not None:
            (
                vsite_values,
                vsite_unfrozen_idxs,
                vsite_scales,
                vsite_clamp_lower,
                vsite_clamp_upper,
            ) = self._prepare_vsites(force_field, vsites)
            self._fit_vsites = True

            values.append(vsite_values)
            unfrozen_idxs.append(
                (vsite_unfrozen_idxs + len(param_scales) + len(attr_scales))
            )
            scales.append(vsite_scales)
            clamp_lower.append(vsite_clamp_lower)
            clamp_upper.append(vsite_clamp_upper)

        self._values = torch.cat(values)

        self._unfrozen_idxs = torch.cat(unfrozen_idxs).long()

        self._scales = torch.cat(scales)[self._unfrozen_idxs]

        self._clamp_lower = torch.cat(clamp_lower)[self._unfrozen_idxs]
        self._clamp_upper = torch.cat(clamp_upper)[self._unfrozen_idxs]

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
        values[self._unfrozen_idxs] = (values_flat / self._scales).clamp(
            min=self._clamp_lower, max=self._clamp_upper
        )
        shapes = self._param_shapes + self._attr_shapes

        if self._fit_vsites:
            shapes.append(self._force_field.v_sites.parameters.shape)

        values = _unflatten_tensors(values, shapes)

        params = values[: len(self._param_shapes)]

        for potential_type, param in zip(self._param_types, params, strict=True):
            potentials[potential_type].parameters = param

        attrs = values[
            len(self._param_shapes) : len(self._param_shapes) + len(self._attr_shapes)
        ]

        for potential_type, attr in zip(self._attr_types, attrs, strict=True):
            potentials[potential_type].attributes = attr

        if self._fit_vsites:
            vsite_params = values[len(self._param_shapes) + len(self._attr_shapes) :]
            self._force_field.v_sites.parameters = vsite_params[0]

        return self._force_field

    @torch.no_grad()
    def clamp(self, values_flat: torch.Tensor) -> torch.Tensor:
        """Clamps the given values to the configured min and max values."""
        return (values_flat / self._scales).clamp(
            min=self._clamp_lower, max=self._clamp_upper
        ) * self._scales
