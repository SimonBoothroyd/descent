import pytest
import torch

from descent import transforms
from descent.transforms import transform_tensor


def test_identity():

    value = torch.rand(4)
    output = transforms.identity()(value)

    assert output.shape == value.shape
    assert torch.allclose(output, value)


@pytest.mark.parametrize(
    "index, expected",
    [(0, torch.tensor([0.0, 1.0, 2.0])), (1, torch.tensor([-1.0, 0.0, 1.0]))],
)
def test_relative(index, expected):

    value = torch.tensor([1.0, 2.0, 3.0])
    output = transforms.relative(index=index)(value)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize(
    "transforms_to_apply, expected",
    [
        (transforms.relative(0), torch.tensor([0.0, 1.0, 2.0])),
        ([], torch.tensor([1.0, 2.0, 3.0])),
        (
            [transforms.relative(0), transforms.relative(1)],
            torch.tensor([-1.0, 0.0, 1.0]),
        ),
    ],
)
def test_transform_tensor(transforms_to_apply, expected):

    value = torch.tensor([1.0, 2.0, 3.0])
    output = transform_tensor(value, transforms_to_apply)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)
