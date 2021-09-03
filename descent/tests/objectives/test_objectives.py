from openff.toolkit.typing.engines.smirnoff import ForceField

from descent.models.smirnoff import SMIRNOFFModel
from descent.objectives import ObjectiveContribution


def test_call(monkeypatch):

    evaluate_called = False

    class DummyObjective(ObjectiveContribution):
        @property
        def parameter_ids(self):
            return []

        def evaluate(self, model):

            nonlocal evaluate_called
            evaluate_called = True

    DummyObjective()(SMIRNOFFModel([], ForceField))
    assert evaluate_called
