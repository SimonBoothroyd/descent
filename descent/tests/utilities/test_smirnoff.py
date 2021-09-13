import pytest
import torch
from openff.interchange.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import unit as simtk_unit

from descent.data import DatasetEntry
from descent.tests import is_close
from descent.utilities.smirnoff import exercised_parameters, perturb_force_field


def test_perturb_force_field():

    smirks = "[*:1]~[*:2]~[*:3]~[*:4]"

    initial_force_field = ForceField()
    initial_force_field.get_parameter_handler("ProperTorsions")

    initial_force_field["ProperTorsions"].add_parameter(
        {
            "smirks": smirks,
            "k": [0.1, 0.2] * simtk_unit.kilocalories_per_mole,
            "phase": [0.0, 180.0] * simtk_unit.degree,
            "periodicity": [1, 2],
            "idivf": [2.0, 1.0],
        }
    )

    perturbed_force_field = perturb_force_field(
        initial_force_field,
        torch.tensor([0.1, 0.2]),
        [
            ("ProperTorsions", PotentialKey(id=smirks, mult=0), "k"),
            ("ProperTorsions", PotentialKey(id=smirks, mult=1), "idivf"),
        ],
    )

    assert is_close(
        perturbed_force_field["ProperTorsions"].parameters[smirks].k1,
        0.1 * simtk_unit.kilocalories_per_mole + 0.1 * simtk_unit.kilojoules_per_mole,
    )
    assert is_close(
        perturbed_force_field["ProperTorsions"].parameters[smirks].idivf2, 1.2
    )


@pytest.mark.parametrize(
    "handlers_to_include,"
    "handlers_to_exclude,"
    "ids_to_include,"
    "ids_to_exclude,"
    "attributes_to_include,"
    "attributes_to_exclude,"
    "n_expected,"
    "expected_handlers,"
    "expected_potential_keys,"
    "expected_attributes",
    [
        (
            None,
            None,
            None,
            None,
            None,
            None,
            36,
            {"Bonds", "Angles"},
            {
                PotentialKey(id=smirks, mult=mult, associated_handler=handler)
                for smirks in ("a", "b", "c")
                for mult in (None, 0, 1)
                for handler in ("Bonds", "Angles")
            },
            {"k", "length", "angle"},
        ),
        (
            ["Bonds"],
            None,
            None,
            None,
            None,
            None,
            18,
            {"Bonds"},
            {
                PotentialKey(id=smirks, mult=mult, associated_handler="Bonds")
                for smirks in ("a", "b", "c")
                for mult in (None, 0, 1)
            },
            {"k", "length"},
        ),
        (
            None,
            ["Bonds"],
            None,
            None,
            None,
            None,
            18,
            {"Angles"},
            {
                PotentialKey(id=smirks, mult=mult, associated_handler="Angles")
                for smirks in ("a", "b", "c")
                for mult in (None, 0, 1)
            },
            {"k", "angle"},
        ),
        (
            None,
            None,
            [PotentialKey(id="b", mult=0, associated_handler="Bonds")],
            None,
            None,
            None,
            2,
            {"Bonds"},
            {PotentialKey(id="b", mult=0, associated_handler="Bonds")},
            {"k", "length"},
        ),
        (
            None,
            None,
            None,
            [
                PotentialKey(id="b", mult=0, associated_handler="Bonds"),
            ],
            None,
            None,
            34,
            {"Bonds", "Angles"},
            {
                PotentialKey(id=smirks, mult=mult, associated_handler=handler)
                for handler in ("Bonds", "Angles")
                for smirks in ("a", "b", "c")
                for mult in (None, 0, 1)
                if (smirks != "b" or mult != 0 or handler != "Bonds")
            },
            {"k", "length", "angle"},
        ),
        (
            None,
            None,
            None,
            None,
            ["length"],
            None,
            9,
            {"Bonds"},
            {
                PotentialKey(id=smirks, mult=mult, associated_handler="Bonds")
                for smirks in ("a", "b", "c")
                for mult in (None, 0, 1)
            },
            {"length"},
        ),
        (
            None,
            None,
            None,
            None,
            None,
            ["length"],
            27,
            {"Bonds", "Angles"},
            {
                PotentialKey(id=smirks, mult=mult, associated_handler=handler)
                for handler in ("Bonds", "Angles")
                for smirks in ("a", "b", "c")
                for mult in (None, 0, 1)
            },
            {"k", "angle"},
        ),
    ],
)
def test_exercised_parameters(
    handlers_to_include,
    handlers_to_exclude,
    ids_to_include,
    ids_to_exclude,
    attributes_to_include,
    attributes_to_exclude,
    n_expected,
    expected_handlers,
    expected_potential_keys,
    expected_attributes,
):
    class MockEntry(DatasetEntry):
        def evaluate_loss(self, model, **kwargs):
            pass

    def mock_entry(handler, patterns, mult):

        attributes = {"Bonds": ["k", "length"], "Angles": ["k", "angle"]}[handler]

        entry = MockEntry.__new__(MockEntry)
        entry._model_input = {
            (handler, ""): (
                None,
                None,
                [
                    (
                        PotentialKey(id=smirks, mult=mult, associated_handler=handler),
                        attributes,
                    )
                    for smirks in patterns
                ],
            )
        }
        return entry

    entries = [
        mock_entry(handler, patterns, mult)
        for handler in ["Bonds", "Angles"]
        for patterns in [("a", "b"), ("b", "c")]
        for mult in [None, 0, 1]
    ]

    parameter_keys = exercised_parameters(
        entries,
        handlers_to_include,
        handlers_to_exclude,
        ids_to_include,
        ids_to_exclude,
        attributes_to_include,
        attributes_to_exclude,
    )

    assert len(parameter_keys) == n_expected

    actual_handlers, actual_keys, actual_attributes = zip(*parameter_keys)

    assert {*actual_handlers} == expected_handlers
    assert {*actual_keys} == expected_potential_keys
    assert {*actual_attributes} == expected_attributes
