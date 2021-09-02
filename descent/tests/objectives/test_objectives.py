import torch

from descent.objectives import ObjectiveContribution


def test_call(monkeypatch):

    evaluate_called = False

    class DummyObjective(ObjectiveContribution):

        @property
        def parameter_ids(self):
            return []

        def evaluate(self, parameter_delta, parameter_delta_ids):

            nonlocal evaluate_called
            evaluate_called = True

    DummyObjective()(torch.tensor([]), [])
    assert evaluate_called
