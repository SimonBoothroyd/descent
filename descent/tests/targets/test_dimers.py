import openff.interchange
import openff.toolkit
import openff.units
import pytest
import smee.converters
import torch

from descent.targets.dimers import (
    Dimer,
    compute_dimer_energy,
    create_dataset,
    create_from_des,
    extract_smiles,
    predict,
    report,
)


@pytest.fixture
def mock_dimer() -> Dimer:
    return {
        "smiles_a": "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
        "smiles_b": "[O:1]([H:2])[H:3]",
        "coords": torch.arange(54, dtype=torch.float32).reshape(2, 9, 3),
        "energy": 3.0 * torch.arange(2, dtype=torch.float32),
        "source": "some source...",
    }


def test_create_dataset(mock_dimer):
    expected_data_entries = [
        {
            "smiles_a": mock_dimer["smiles_a"],
            "smiles_b": mock_dimer["smiles_b"],
            "coords": mock_dimer["coords"].flatten().tolist(),
            "energy": mock_dimer["energy"].tolist(),
            "source": mock_dimer["source"],
        },
    ]

    dataset = create_dataset([mock_dimer])
    assert len(dataset) == 1

    data_entries = dataset.to_pylist()

    assert data_entries == pytest.approx(expected_data_entries)


def test_create_from_des(data_dir):
    expected_coords = torch.arange(6 * 3 + 3 * 3, dtype=torch.float32).reshape(1, 9, 3)

    def energy_fn(data, ids, coords):
        assert coords.shape == expected_coords.shape
        assert torch.allclose(coords, expected_coords)

        assert ids == (123,)

        return torch.tensor(data["reference"].values)

    dataset = create_from_des(data_dir / "DESMOCK", energy_fn)
    assert len(dataset) == 1

    expected = {
        "smiles_a": "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
        "smiles_b": "[O:1]([H:2])[H:3]",
        "coords": expected_coords.flatten().tolist(),
        "energy": [-1.23],
        "source": "DESMOCK system=4321 orig=MOCK group=1423",
    }

    assert dataset.to_pylist() == [pytest.approx(expected)]


def test_extract_smiles(mock_dimer):
    expected_smiles = ["[C:1]([O:2][H:6])([H:3])([H:4])[H:5]", "[O:1]([H:2])[H:3]"]

    dataset = create_dataset([mock_dimer, mock_dimer])
    smiles = extract_smiles(dataset)

    assert smiles == expected_smiles


def test_compute_dimer_energy():
    openff_ff = openff.toolkit.ForceField()
    openff_ff.get_parameter_handler("vdW").add_parameter(
        {
            "smirks": "[Ar:1]",
            "epsilon": 1.0 * openff.units.unit.kilocalorie / openff.units.unit.mole,
            "sigma": 1.0 * openff.units.unit.angstrom,
        }
    )
    openff_ff.get_parameter_handler("vdW").add_parameter(
        {
            "smirks": "[He:1]",
            "epsilon": 4.0 * openff.units.unit.kilocalorie / openff.units.unit.mole,
            "sigma": 1.0 * openff.units.unit.angstrom,
        }
    )

    interchanges = [
        openff.interchange.Interchange.from_smirnoff(
            openff_ff, openff.toolkit.Molecule.from_smiles(smiles).to_topology()
        )
        for smiles in ("[Ar]", "[He]")
    ]
    tensor_ff, [top_a, top_b] = smee.converters.convert_interchange(interchanges)

    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )

    # eps = sqrt(4 * 1)
    expected_energies = (
        4.0 * 2.0 * torch.tensor([0.0, (1.0 / 2.0) ** 12 - (1.0 / 2.0) ** 6])
    )

    energies = compute_dimer_energy(top_a, top_b, tensor_ff, coords)
    assert energies.shape == expected_energies.shape
    assert torch.allclose(energies, expected_energies)


def test_compute_dimer_energy_v_sites():
    openff_ff = openff.toolkit.ForceField("tip4p_fb.offxml")

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff_ff, openff.toolkit.Molecule.from_smiles("O").to_topology()
    )
    tensor_ff, [top] = smee.converters.convert_interchange(interchange)

    coords = torch.tensor(
        [
            [
                [-1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )

    energies = compute_dimer_energy(top, top, tensor_ff, coords)
    assert energies.shape == (1,)
    assert not torch.isnan(energies).any()


def test_predict(mock_dimer, mocker):
    dataset = create_dataset([mock_dimer])

    expected_y_pred = torch.Tensor([-1.23, 4.56])

    mock_energy_fn = mocker.patch(
        "descent.targets.dimers.compute_dimer_energy",
        autospec=True,
        return_value=expected_y_pred,
    )

    mock_ff = mocker.MagicMock()
    mock_ff.potentials[0].parameters = torch.zeros(1)

    mock_top_a = mocker.Mock()
    mock_tob_b = mocker.Mock()

    topologies = {
        mock_dimer["smiles_a"]: mock_top_a,
        mock_dimer["smiles_b"]: mock_tob_b,
    }

    y_ref, y_pred = predict(dataset, mock_ff, topologies)

    assert y_pred.shape == (2,)
    assert torch.allclose(y_pred, expected_y_pred)

    assert y_ref.shape == mock_dimer["energy"].shape
    assert torch.allclose(y_ref, mock_dimer["energy"])

    expected_coords = mock_dimer["coords"]

    mock_energy_fn.assert_called_once_with(
        mock_top_a, mock_tob_b, mock_ff, pytest.approx(expected_coords)
    )


def test_report(tmp_cwd, mock_dimer, mocker):
    dataset = create_dataset([mock_dimer])

    expected_y_pred = torch.Tensor([-1.23, 4.56])

    mock_predict_fn = mocker.patch(
        "descent.targets.dimers._predict",
        autospec=True,
        return_value=(None, expected_y_pred),
    )

    mock_ff = mocker.MagicMock()
    mock_tops = mocker.MagicMock()

    expected_path = tmp_cwd / "report.html"
    report(dataset, {"A": mock_ff}, mock_tops, expected_path)

    assert expected_path.exists()
    assert expected_path.read_text().startswith("<style>.itable")

    mock_predict_fn.assert_called_once_with(mocker.ANY, mock_ff, mock_tops)
