from typing import Callable, Union

import numpy
from openff.toolkit.utils import string_to_unit
from openff.units import unit
from openff.units.simtk import unit_to_string
from simtk import unit as simtk_unit


def _compare_values(
    a: Union[float, unit.Quantity, simtk_unit.Quantity],
    b: Union[float, unit.Quantity, simtk_unit.Quantity],
    predicate: Callable[
        [Union[float, numpy.ndarray], Union[float, numpy.ndarray]], bool
    ],
) -> bool:
    """Compare to values using a specified predicate taking units into account."""

    if isinstance(a, simtk_unit.Quantity):

        expected_unit = unit.Unit(unit_to_string(a.unit))
        a = a.value_in_unit(a.unit)

    elif isinstance(a, unit.Quantity):

        expected_unit = a.units
        a = a.to(expected_unit).magnitude

    else:

        expected_unit = None

    if isinstance(b, simtk_unit.Quantity):

        assert expected_unit is not None, "cannot compare quantity with unit-less."
        b = b.value_in_unit(string_to_unit(f"{expected_unit:!s}"))

    elif isinstance(b, unit.Quantity):

        assert expected_unit is not None, "cannot compare quantity with unit-less."
        b = b.to(expected_unit).magnitude

    else:

        assert expected_unit is None, "cannot compare quantity with unit-less."

    return predicate(a, b)


def is_close(
    a: Union[float, unit.Quantity, simtk_unit.Quantity],
    b: Union[float, unit.Quantity, simtk_unit.Quantity],
) -> bool:
    """Compare whether two values are close taking units into account."""

    return _compare_values(a, b, numpy.isclose)


def all_close(
    a: Union[numpy.ndarray, unit.Quantity, simtk_unit.Quantity],
    b: Union[numpy.ndarray, unit.Quantity, simtk_unit.Quantity],
) -> bool:
    """Compare whether all elements in two array are close taking units into account."""

    return _compare_values(a, b, numpy.allclose)
