import copy
from typing import List, Tuple

import torch
from openff.system.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils import string_to_unit
from smirnoffee.smirnoff import _DEFAULT_UNITS


def perturb_force_field(
    force_field: ForceField,
    parameter_delta: torch.Tensor,
    parameter_delta_ids: List[Tuple[str, PotentialKey, str]],
) -> ForceField:
    """Perturbs the specified parameters in a force field by the provided delta values.

    Args:
        force_field: The force field to perturb.
        parameter_delta: A 1D tensor of deltas to add to the parameters referenced by
            the ``parameter_delta_ids``.
        parameter_delta_ids:
            The unique identifiers which maps each value in the ``parameter_delta``
            tensor to a SMIRNOFF force field parameter. These should be of the form:
            ``(handler_name, potential_key, attribute_name)``.

    Returns:
        The perturbed force field.
    """
    from simtk import unit as simtk_unit

    force_field = copy.deepcopy(force_field)

    for (handler_name, potential_key, attribute), delta in zip(
        parameter_delta_ids, parameter_delta
    ):

        parameter = force_field[handler_name].parameters[potential_key.id]

        delta = delta.detach().item() * string_to_unit(
            f"{_DEFAULT_UNITS[handler_name][attribute]:!s}"
        )

        if potential_key.mult is not None:
            attribute = f"{attribute}{potential_key.mult + 1}"

        original_value = getattr(parameter, attribute)

        if not isinstance(original_value, simtk_unit.Quantity):
            delta = delta.value_in_unit(simtk_unit.dimensionless)

        setattr(parameter, attribute, original_value + delta)

    return force_field
