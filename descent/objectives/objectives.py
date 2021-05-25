import abc
from typing import List, Tuple

import torch
from openff.system.models import PotentialKey


class ObjectiveContribution(abc.ABC):
    """The base class for contributions to a total objective function."""

    @property
    @abc.abstractmethod
    def parameter_ids(self) -> List[Tuple[str, PotentialKey, str]]:
        """The ids of the parameters that are exercised by this contribution to the
        total objective function.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(
        self,
        parameter_delta: torch.Tensor,
        parameter_delta_ids: List[Tuple[str, PotentialKey, str]],
    ) -> torch.Tensor:
        """Evaluate the objective at the given parameter offsets"""
        raise NotImplementedError()

    def __call__(
        self,
        parameter_delta: torch.Tensor,
        parameter_delta_ids: List[Tuple[str, PotentialKey, str]],
    ) -> torch.Tensor:
        """Evaluate the objective at the given parameter offsets"""
        return self.evaluate(parameter_delta, parameter_delta_ids)
