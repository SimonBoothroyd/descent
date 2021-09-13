import io
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch.nn
from openff.interchange.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from smirnoffee.potentials import add_parameter_delta
from typing_extensions import Literal

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

    def summarise(
        self,
        parameter_id_type: Literal["smirks", "id"] = "smirks",
        print_to_terminal: bool = True,
    ) -> str:
        """

        Args:
            parameter_id_type: The type of ID to show for each parameter. Currently
                this can either be the unique ``'id'`` associated with the parameter or
                the ``'smirks'`` pattern that encodes the chemical environment the
                parameter is applied to.
            print_to_terminal: Whether to print the summary to the terminal

        Returns:
            A string containing the summary.
        """

        from openff.units.simtk import from_simtk

        final_force_field = self.to_force_field()

        # Reshape the data into dictionaries to make tabulation easier
        table_data = defaultdict(lambda: defaultdict(dict))
        attribute_units = {}

        for handler_type, potential_key, attribute in [
            (handler_type, potential_key, attribute)
            for handler_type, parameter_ids in self._parameter_delta_ids.items()
            for (potential_key, attribute) in parameter_ids
        ]:

            smirks = potential_key.id

            attribute = (
                attribute
                if potential_key.mult is None
                else f"{attribute}{potential_key.mult}"
            )

            initial_value = from_simtk(
                getattr(
                    self._initial_force_field[handler_type].parameters[smirks],
                    attribute,
                )
            )
            final_value = from_simtk(
                getattr(final_force_field[handler_type].parameters[smirks], attribute)
            )

            if (handler_type, attribute) not in attribute_units:
                attribute_units[(handler_type, attribute)] = initial_value.units

            unit = attribute_units[(handler_type, attribute)]

            attribute = f"{attribute} ({unit:P~})"

            if parameter_id_type == "id":

                smirks = self._initial_force_field[handler_type].parameters[smirks].id
                smirks = smirks if smirks is not None else "NO ID"

            table_data[handler_type][attribute][smirks] = (
                initial_value.to(unit).m,
                final_value.to(unit).m,
            )

        # Construct the final return value:
        return_value = io.StringIO()

        for handler_type, attribute_data in table_data.items():

            print(f"\n{handler_type.center(80, '=')}\n", file=return_value)

            attribute_headers = sorted(attribute_data)

            attribute_widths = {
                attribute: max(
                    [
                        len(f"{value:.4f}")
                        for value_tuple in attribute_data[attribute].values()
                        for value in value_tuple
                    ]
                )
                * 2
                + 1
                for attribute in attribute_headers
            }
            attribute_widths = {
                # Make sure the width of the column - 1 is divisible by 2
                attribute: max(int((column_width - 1) / 2.0 + 0.5) * 2 + 1, 15)
                for attribute, column_width in attribute_widths.items()
            }

            smirks_width = max(
                len(smirks)
                for smirks_data in attribute_data.values()
                for smirks in smirks_data
            )

            first_header = (
                " " * (smirks_width)
                + "   "
                + "   ".join(
                    [
                        attribute.center(attribute_widths[attribute], " ")
                        for attribute in attribute_headers
                    ]
                )
            )
            second_header = (
                " " * (smirks_width)
                + "   "
                + "   ".join(
                    [
                        "INITIAL".center((column_width - 1) // 2, " ")
                        + " "
                        + "FINAL".center((column_width - 1) // 2, " ")
                        for attribute, column_width in attribute_widths.items()
                    ]
                )
            )
            border = (
                "-" * smirks_width
                + "   "
                + "   ".join(
                    [
                        "-" * attribute_widths[attribute]
                        for attribute in attribute_headers
                    ]
                )
            )

            smirks_data = defaultdict(dict)

            for attribute in attribute_data:
                for smirks, value_tuple in attribute_data[attribute].items():
                    smirks_data[smirks][attribute] = value_tuple

            print(border, file=return_value)
            print(first_header, file=return_value)
            print(second_header, file=return_value)
            print(border, file=return_value)

            for smirks in sorted(smirks_data):

                def format_column(attr, value_tuple):

                    if value_tuple is None:
                        return " " * attribute_widths[attr]

                    value_width = (attribute_widths[attr] - 1) // 2
                    return (
                        f"{value_tuple[0]:.4f}".ljust(value_width, " ")
                        + " "
                        + f"{value_tuple[1]:.4f}".ljust(value_width, " ")
                    )

                row = (
                    f"{smirks.ljust(smirks_width)}"
                    + "   "
                    + "   ".join(
                        [
                            format_column(
                                attribute, smirks_data[smirks].get(attribute, None)
                            )
                            for attribute in attribute_headers
                        ]
                    )
                )

                print(row, file=return_value)

        return_value = return_value.getvalue()

        if print_to_terminal:
            print(return_value)

        return return_value
