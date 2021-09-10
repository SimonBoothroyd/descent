import abc
from typing import Generic, Iterator, Sequence, TypeVar, Union

import torch.utils.data
from openff.interchange.components.interchange import Interchange
from smirnoffee.smirnoff import vectorize_system

from descent.models import ParameterizationModel
from descent.models.models import VectorizedSystem

T_co = TypeVar("T_co", covariant=True)


class DatasetEntry(abc.ABC):
    """The base class for storing labels associated with an input datum, such as
    an OpenFF interchange object or an Espaloma graph model."""

    @property
    def model_input(self) -> VectorizedSystem:
        return self._model_input

    def __init__(self, model_input: Union[Interchange]):
        """

        Args:
            model_input: The input that will be passed to the model being trained in
                order to yield a vectorized view of a parameterised molecule. If the
                input is an interchange object it will be vectorised prior to being
                used as a model input.
        """

        self._model_input = (
            model_input
            if not isinstance(model_input, Interchange)
            else vectorize_system(model_input)
        )

    @abc.abstractmethod
    def evaluate(self, model: ParameterizationModel, **kwargs) -> torch.Tensor:
        """Evaluates the contribution to the total loss function of the data stored
        in this entry using a specified model.

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.

        Returns:
            The loss contribution of this entry.
        """
        raise NotImplementedError()

    def __call__(self, model: ParameterizationModel, **kwargs) -> torch.Tensor:
        """Evaluate the objective using a specified model.

        Args:
            model: The model that will return vectorized view of a parameterised
                molecule.

        Returns:
            The loss contribution of this entry.
        """
        return self.evaluate(model, **kwargs)


class Dataset(torch.utils.data.IterableDataset[T_co], Generic[T_co]):
    r"""An class representing a :class:`Dataset`."""

    def __init__(self, entries: Sequence):
        self._entries = entries

    def __getitem__(self, index: int) -> T_co:
        return self._entries[index]

    def __iter__(self) -> Iterator[T_co]:
        return self._entries.__iter__()

    def __len__(self) -> int:
        return len(self._entries)
