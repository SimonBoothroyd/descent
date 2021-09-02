import abc
from typing import List, Optional, Tuple

import torch
from openff.interchange.models import PotentialKey


class ObjectiveContribution(abc.ABC):
    """The base class for contributions to a total objective ( / loss) function."""

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
        parameter_delta: Optional[torch.Tensor],
        parameter_delta_ids: Optional[List[Tuple[str, PotentialKey, str]]],
    ) -> torch.Tensor:
        """Evaluate the objective at the given parameter offsets

        Args:
            parameter_delta: An optional tensor of values to perturb the assigned
                parameters by before evaluating the potential energy.
            parameter_delta_ids: An optional list of ids associated with the
                ``parameter_delta`` tensor which is used to identify which parameter
                delta matches which assigned parameter.

        Returns:
            The loss contribution of this term.
        """
        raise NotImplementedError()

    def __call__(
        self,
        parameter_delta: Optional[torch.Tensor],
        parameter_delta_ids: Optional[List[Tuple[str, PotentialKey, str]]],
    ) -> torch.Tensor:
        """Evaluate the objective at the given parameter offsets

        Args:
            parameter_delta: An optional tensor of values to perturb the assigned
                parameters by before evaluating the potential energy.
            parameter_delta_ids: An optional list of ids associated with the
                ``parameter_delta`` tensor which is used to identify which parameter
                delta matches which assigned parameter.

        Returns:
            The loss contribution of this term.
        """
        return self.evaluate(parameter_delta, parameter_delta_ids)
