import copy
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import torch
from openff.interchange.components.interchange import Interchange
from openff.interchange.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils import string_to_unit
from smirnoffee.smirnoff import _DEFAULT_UNITS, vectorize_system

from descent.utilities import value_or_list_to_list

if TYPE_CHECKING:
    from descent.data import Dataset, DatasetEntry


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


def exercised_parameters(
    dataset: Union["Dataset", Iterable["DatasetEntry"], Iterable[Interchange]],
    handlers_to_include: Optional[Union[str, List[str]]] = None,
    handlers_to_exclude: Optional[Union[str, List[str]]] = None,
    ids_to_include: Optional[Union[PotentialKey, List[PotentialKey]]] = None,
    ids_to_exclude: Optional[Union[PotentialKey, List[PotentialKey]]] = None,
    attributes_to_include: Optional[Union[str, List[str]]] = None,
    attributes_to_exclude: Optional[Union[str, List[str]]] = None,
) -> List[Tuple[str, PotentialKey, str]]:
    """Returns the identifiers of each parameter that has been assigned to each molecule
    in a dataset.

    Notes:
        This function assumes that the dataset was created using an OpenFF interchange
        object as the main input.

    Args:
        dataset: The dataset, list of dataset entries, or list of interchange objects
            That track a set of SMIRNOFF parameters assigned to a set of molecules.
        handlers_to_include: An optional list of the parameter handlers that the returned
            parameters should be associated with.
        handlers_to_exclude: An optional list of the parameter handlers that the returned
            parameters should **not** be associated with.
        ids_to_include: An optional list of the potential keys that the parameters should
            match with to be returned.
        ids_to_exclude: An optional list of the potential keys that the parameters should
            **not** match with to be returned.
        attributes_to_include: An optional list of the attributes that the parameters
            should match with to be returned.
        attributes_to_exclude: An optional list of the attributes that the parameters
            should **not** match with to be returned.

    Returns:
        A list of tuples of the form ``(handler_type, potential_key, attribute_name)``.
    """

    def should_skip(value, to_include, to_exclude) -> bool:

        to_include = value_or_list_to_list(to_include)
        to_exclude = value_or_list_to_list(to_exclude)

        return (to_include is not None and value not in to_include) or (
            to_exclude is not None and value in to_exclude
        )

    vectorized_systems = [
        entry.model_input
        if not isinstance(entry, Interchange)
        else vectorize_system(entry)
        for entry in dataset
    ]

    return_value = {
        (handler_type, potential_key, attribute)
        for vectorized_system in vectorized_systems
        for (handler_type, _), (*_, potential_keys) in vectorized_system.items()
        if not should_skip(handler_type, handlers_to_include, handlers_to_exclude)
        for (potential_key, attributes) in potential_keys
        if not should_skip(potential_key, ids_to_include, ids_to_exclude)
        for attribute in attributes
        if not should_skip(attribute, attributes_to_include, attributes_to_exclude)
    }

    return_value = sorted(
        return_value,
        key=lambda x: (x[0], x[1].id, x[1].mult if x[1].mult is not None else -1, x[2]),
    )

    return return_value
