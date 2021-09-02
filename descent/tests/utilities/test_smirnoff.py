import torch
from openff.interchange.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import unit as simtk_unit

from descent.tests import is_close
from descent.utilities.smirnoff import perturb_force_field


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
