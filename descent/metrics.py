"""Common and composable loss metrics."""
from typing import Callable, Tuple, Union

import torch

LossMetric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mse(dim: Union[int, Tuple[int, ...]] = None) -> LossMetric:

    if dim is None:
        dim = ()

    def _mse(input_tensor: torch.Tensor, reference_tensor: torch.Tensor):
        return (input_tensor - reference_tensor).square().mean(dim=dim)

    return _mse
