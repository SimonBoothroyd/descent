import pytest

from descent.data import Dataset, DatasetEntry
from descent.models.smirnoff import SMIRNOFFModel
from descent.tests.mocking.systems import generate_mock_hcl_system


class DummyEntry(DatasetEntry):
    def evaluate_loss(self, model, **kwargs):
        pass


class DummyDataset(Dataset[DummyEntry]):
    pass


def test_call(monkeypatch):

    evaluate_called = False
    evaluate_kwargs = {}

    class LocalEntry(DatasetEntry):
        def evaluate_loss(self, model, **kwargs):
            nonlocal evaluate_called
            evaluate_called = True
            evaluate_kwargs.update(kwargs)

    LocalEntry(generate_mock_hcl_system())(SMIRNOFFModel([], None), a="a", b=2)

    assert evaluate_called
    assert evaluate_kwargs == {"a": "a", "b": 2}


def test_dataset():

    model_input = generate_mock_hcl_system()

    dataset = DummyDataset(entries=[DummyEntry(model_input), DummyEntry(model_input)])

    assert dataset[0] is not None
    assert dataset[1] is not None

    with pytest.raises(IndexError):
        assert dataset[2]

    assert len(dataset) == 2

    assert all(isinstance(entry.model_input, dict) for entry in dataset)
