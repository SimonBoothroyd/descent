from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch.nn
from openff.interchange.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from smirnoffee.potentials import add_parameter_delta

from descent.models.models import VectorizedSystem
from descent.utilities.smirnoff import perturb_force_field


class SMIRNOFFModel(torch.nn.Module):
    """A model for perturbing a set of SMIRNOFF parameters that have been applied to a
    models via the ``openff-interchange`` package.
    """

    @property
    def parameter_delta_ids(self) -> Tuple[Tuple[str, PotentialKey, str], ...]:
        """
        Returns:
            The 'ids' of the force field parameters that will be perturbed by this model
            where each 'id' is a tuple of ``(handler_type, smirks, attribute)``.
        """

        return tuple(
            (handler_type, smirks, attribute)
            for handler_type, parameter_ids in self._parameter_delta_ids.items()
            for (smirks, attribute) in parameter_ids
        )

    def __init__(
        self,
        parameter_ids: List[Tuple[str, Union[str, PotentialKey], str]],
        initial_force_field: Optional[ForceField],
        covariance_tensor: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            parameter_ids: A list of the 'ids' of the parameters that will be applied
                where each id should be a tuple of ``(handler_type, smirks, attribute)``.
            initial_force_field: The (optional) force field used to initially
                parameterize the molecules of interest.
            covariance_tensor: A tensor that will be used to transform an the
                ``parameter_delta`` before it is used by the ``forward`` method such
                that ``parameter_delta = self.covariance_tensor @ self.parameter_delta``.
                It should have shape=``(n_parameter_ids, n_hidden)`` where ``n_hidden``
                is the number of parameters prior to applying the transform.

                This can be used to scale the values of parameters:

                ``covariance_tensor = torch.eye(len(parameter_ids)) * 0.01``

                or even define the covariance between parameters as in the case of BCCs:

                ``covariance_tensor = torch.tensor([[1.0], [-1.0]])``

                Usually ``n_hidden <= n_parameter_ids``.
        """

        super(SMIRNOFFModel, self).__init__()

        self._initial_force_field = initial_force_field

        self._parameter_delta_ids = defaultdict(list)

        for handler_type, smirks, attribute in parameter_ids:

            self._parameter_delta_ids[handler_type].append(
                (
                    smirks
                    if isinstance(smirks, PotentialKey)
                    else PotentialKey(id=smirks, associated_handler=handler_type),
                    attribute,
                )
            )

        # Convert the ids to a normal dictionary to make sure KeyErrors get raised again.
        self._parameter_delta_ids: Dict[str, List[Tuple[PotentialKey, str]]] = {
            **self._parameter_delta_ids
        }

        self._parameter_delta_indices = {}
        counter = 0

        for handler_type, handler_ids in self._parameter_delta_ids.items():

            self._parameter_delta_indices[handler_type] = torch.arange(
                counter, counter + len(handler_ids), dtype=torch.int64
            )

            counter += len(handler_ids)

        self._covariance_tensor = covariance_tensor

        assert covariance_tensor is None or (
            covariance_tensor.ndim == 2
            and covariance_tensor.shape[0] == len(parameter_ids)
        ), "the ``covariance_tensor`` must have shape=``(n_parameter_ids, n_hidden)``"

        n_parameter_deltas = (
            len(parameter_ids) if covariance_tensor is None else len(covariance_tensor)
        )

        self.parameter_delta = torch.nn.Parameter(
            torch.zeros(n_parameter_deltas).requires_grad_(), requires_grad=True
        )

    def forward(self, graph: VectorizedSystem) -> VectorizedSystem:
        """Perturb the parameters of an already applied force field using this models
        current parameter 'deltas'.
        """

        if len(self._parameter_delta_ids) == 0:
            return graph

        parameter_delta = self.parameter_delta

        if self._covariance_tensor:
            parameter_delta = self._covariance_tensor @ parameter_delta

        output = {}

        for (handler_type, handler_expression), vectorized_handler in graph.items():

            handler_delta_ids = self._parameter_delta_ids.get(handler_type, [])

            if len(handler_delta_ids) == 0:

                output[(handler_type, handler_expression)] = vectorized_handler
                continue

            handler_delta_indices = self._parameter_delta_indices[handler_type]
            handler_delta = parameter_delta[handler_delta_indices]

            indices, handler_parameters, handler_parameter_ids = vectorized_handler

            perturbed_parameters = add_parameter_delta(
                handler_parameters,
                handler_parameter_ids,
                handler_delta,
                handler_delta_ids,
            )

            output[(handler_type, handler_expression)] = (
                indices,
                perturbed_parameters,
                handler_parameter_ids,
            )

        return output

    def to_force_field(self) -> ForceField:
        """Returns the current force field (i.e. initial_force_field + parameter_delta)
        as an OpenFF force field object.
        """

        return perturb_force_field(
            self._initial_force_field,
            self.parameter_delta
            if self._covariance_tensor is None
            else self._covariance_tensor @ self.parameter_delta,
            [
                (handler_type, smirks, attribute)
                for handler_type, handler_ids in self._parameter_delta_ids.items()
                for (smirks, attribute) in handler_ids
            ],
        )
