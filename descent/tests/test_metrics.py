import pytest
import torch

from descent import metrics


@pytest.mark.parametrize(
    "dim, expected",
    [
        (0, torch.tensor([13.0 / 2.0, 5.0 / 2.0])),
        (1, torch.tensor([10.0 / 2.0, 8.0 / 2.0])),
        ((), torch.tensor(18.0 / 4.0)),
    ],
)
def test_mse(dim, expected):

    # 3 1
    # 2 2

    input_a = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    input_b = torch.tensor([[4.0, 3.0], [7.0, 8.0]])

    output = metrics.mse(dim=dim)(input_a, input_b)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)
