import openff.interchange
import openff.toolkit
import pytest
import smee.converters
import torch

import descent.utils.dataset
from descent.targets.energy import Entry, create_dataset, extract_smiles, predict


@pytest.fixture
def mock_meoh_entry() -> Entry:
    return {
        "smiles": "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
        "coords": torch.arange(36, dtype=torch.float32).reshape(2, 6, 3),
        "energy": 3.0 * torch.arange(2, dtype=torch.float32),
        "forces": torch.arange(36, dtype=torch.float32).reshape(2, 6, 3) + 36.0,
    }


@pytest.fixture
def mock_hoh_entry() -> Entry:
    return {
        "smiles": "[H:2][O:1][H:3]",
        "coords": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [-1.0, -0.5, 0.0], [1.0, -0.5, 0.0]],
                [[0.0, 0.0, 0.0], [-0.7, -0.5, 0.0], [0.7, -0.5, 0.0]],
            ]
        ),
        "energy": torch.tensor([2.0, 3.0]),
        "forces": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
    }


def test_create_dataset(mock_meoh_entry):
    expected_entries = [
        {
            "smiles": mock_meoh_entry["smiles"],
            "coords": pytest.approx(mock_meoh_entry["coords"].flatten()),
            "energy": pytest.approx(mock_meoh_entry["energy"]),
            "forces": pytest.approx(mock_meoh_entry["forces"].flatten()),
        },
    ]

    dataset = create_dataset([mock_meoh_entry])
    assert len(dataset) == 1

    entries = list(descent.utils.dataset.iter_dataset(dataset))
    assert entries == expected_entries


def test_extract_smiles(mock_meoh_entry, mock_hoh_entry):
    expected_smiles = ["[C:1]([O:2][H:6])([H:3])([H:4])[H:5]", "[H:2][O:1][H:3]"]

    dataset = create_dataset([mock_meoh_entry, mock_hoh_entry])
    smiles = extract_smiles(dataset)

    assert smiles == expected_smiles


@pytest.mark.parametrize(
    "reference, "
    "expected_energy_ref, expected_forces_ref, "
    "expected_energy_pred, expected_forces_pred",
    [
        (
            "mean",
            torch.tensor([-0.5, 0.5]),
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ]
            ),
            torch.tensor([7.899425506591797, -7.89942741394043]),
            torch.tensor(
                [
                    [0.0, 83.55978393554688, 0.0],
                    [-161.40325927734375, -41.77988815307617, 0.0],
                    [161.40325927734375, -41.77988815307617, 0.0],
                    [0.0, -137.45770263671875, 0.0],
                    [102.62999725341797, 68.72884368896484, 0.0],
                    [-102.62999725341797, 68.72884368896484, 0.0],
                ]
            ),
        ),
        (
            "min",
            torch.tensor([0.0, 1.0]),
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ]
            ),
            torch.tensor([0.0, -15.798852920532227]),
            torch.tensor(
                [
                    [0.0, 83.55978393554688, 0.0],
                    [-161.40325927734375, -41.77988815307617, 0.0],
                    [161.40325927734375, -41.77988815307617, 0.0],
                    [0.0, -137.45770263671875, 0.0],
                    [102.62999725341797, 68.72884368896484, 0.0],
                    [-102.62999725341797, 68.72884368896484, 0.0],
                ]
            ),
        ),
    ],
)
def test_predict(
    reference,
    expected_energy_ref,
    expected_forces_ref,
    expected_energy_pred,
    expected_forces_pred,
    mock_hoh_entry,
):
    dataset = create_dataset([mock_hoh_entry])

    force_field, [topology] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField("openff-1.3.0.offxml"),
            openff.toolkit.Molecule.from_mapped_smiles(
                mock_hoh_entry["smiles"]
            ).to_topology(),
        )
    )
    topologies = {mock_hoh_entry["smiles"]: topology}

    energy_ref, energy_pred, forces_ref, forces_pred = predict(
        dataset, force_field, topologies, reference=reference
    )

    assert energy_pred.shape == expected_energy_pred.shape
    assert torch.allclose(energy_pred, expected_energy_pred)
    assert energy_ref.shape == expected_energy_ref.shape
    assert torch.allclose(energy_ref, expected_energy_ref)

    assert forces_pred.shape == expected_forces_pred.shape
    assert torch.allclose(forces_pred, expected_forces_pred)
    assert forces_ref.shape == expected_forces_ref.shape
    assert torch.allclose(forces_ref, expected_forces_ref)
