from typing import List, TypeVar, Union, overload

T = TypeVar("T")


@overload
def value_or_list_to_list(value: Union[T, List[T]]) -> List[T]:
    ...


@overload
def value_or_list_to_list(value: None) -> None:
    ...


def value_or_list_to_list(value):

    if value is None:
        return value

    return value if isinstance(value, list) else [value]
