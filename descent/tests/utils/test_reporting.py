import openff.interchange
import openff.toolkit
import smee.converters
from matplotlib import pyplot
from rdkit import Chem

from descent.utils.reporting import (
    _mol_from_smiles,
    figure_to_img,
    mols_to_img,
    print_force_field_summary,
)


def test_mol_from_smiles():
    mol = _mol_from_smiles("[H:2][C:1]([H:3])([H:4])[O:5][H:6]")
    assert Chem.MolToSmiles(mol) == "CO"


def test_mols_to_img():
    img = mols_to_img("CO", "CC")
    assert img.startswith('<img src="data:image/svg+xml;base64,')
    assert img.endswith('"></img>')


def test_figure_to_img():
    figure = pyplot.figure()
    img = figure_to_img(figure)
    pyplot.close(figure)

    assert img.startswith('<img src="data:image/svg+xml;base64,')
    assert img.endswith('"></img>')


def test_print_force_field_summary(capsys):
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("tip4p_fb.offxml"),
        openff.toolkit.Molecule.from_smiles("O").to_topology(),
    )

    force_field, _ = smee.converters.convert_interchange(interchange)
    print_force_field_summary(force_field)

    captured = capsys.readouterr().out

    assert "ID distance [Å] inPlaneAngle [rad] outOfPlaneAngle [rad]" in captured
    assert (
        "[#1:2]-[#8X2H2+0:1]-[#1:3] EP      -0.1053             "
        "3.1416                0.0000" in captured
    )

    assert "fn=4*epsilon*((sigma/r)**12-(sigma/r)**6)" in captured
    assert "scale_12 scale_13 scale_14 scale_15 cutoff [Å] switch_width [Å]" in captured

    assert "ID epsilon [kcal/mol] sigma [Å]" in captured
    assert "[#1:2]-[#8X2H2+0:1]-[#1:3] EP             0.0000    1.0000" in captured
