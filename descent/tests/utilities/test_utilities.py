import pytest

from descent.utilities import value_or_list_to_list


@pytest.mark.parametrize(
    "function_input, expected_output",
    [(None, None), (2, [2]), ("a", ["a"]), ([1, 2], [1, 2]), (["a", "b"], ["a", "b"])],
)
def test_value_or_list_to_list(function_input, expected_output):

    actual_output = value_or_list_to_list(function_input)
    assert expected_output == actual_output
