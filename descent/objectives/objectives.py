import abc
from typing import List, Tuple

import torch
from openff.interchange.models import PotentialKey

from descent.models import ParameterizationModel


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
    def evaluate(self, model: ParameterizationModel) -> torch.Tensor:
        """Evaluate the objective using a specified model.

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.

        Returns:
            The loss contribution of this term.
        """
        raise NotImplementedError()

    def __call__(self, model: ParameterizationModel) -> torch.Tensor:
        """Evaluate the objective using a specified model.

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.

        Returns:
            The loss contribution of this term.
        """
        return self.evaluate(model)
