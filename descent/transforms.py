"""Common and composable tensor transformations useful when computing loss metrics."""
from typing import Callable, Iterable, List, Union

import torch

LossTransform = Callable[[torch.Tensor], torch.Tensor]


def identity() -> LossTransform:
    def _identity(input_tensor: torch.Tensor):
        return input_tensor

    return _identity


def relative(index: int = 0) -> LossTransform:
    def _relative(input_tensor: torch.Tensor):
        return input_tensor - input_tensor[index]

    return _relative


def transform_tensor(
    input_tensor: torch.Tensor, transforms: Union[LossTransform, List[LossTransform]]
) -> torch.Tensor:
    """Applies a set of transforms to an input tensor.

    Args:
        input_tensor: The tensor to transorm.
        transforms: The transforms to apply. If ``None``, the input tensor will be returned.

    Returns:

    """

    if not isinstance(transforms, Iterable):
        transforms = [transforms]

    for transform in transforms:
        input_tensor = transform(input_tensor)

    return input_tensor
