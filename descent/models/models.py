from typing import Any, Dict, Tuple

from smirnoffee.smirnoff import VectorizedHandler
from typing_extensions import Protocol, runtime_checkable

VectorizedSystem = Dict[Tuple[str, str], VectorizedHandler]


@runtime_checkable
class ParameterizationModel(Protocol):
    """The interface the parameterization models must implement."""

    def forward(self, graph: Any) -> VectorizedSystem:
        """Outputs a vectorised view of a parameterized molecule."""

    def summarise(self):
        """Print a summary of the status of this model, such as the differences between
        the initial and current state during training."""
