from typing import List, Union

import numpy
from openff.qcsubmit.results import (
    BasicResult,
    BasicResultCollection,
    OptimizationResult,
    OptimizationResultCollection,
)
from openff.toolkit.topology import Molecule
from pydantic import BaseModel
from qcelemental.models import DriverEnum
from qcportal.models import ObjectId, OptimizationRecord, QCSpecification
from qcportal.models.records import RecordStatusEnum, ResultRecord

DEFAULT_SERVER_ADDRESS = "http://localhost:1234/"


class _FractalClient(BaseModel):

    address: str


def mock_basic_result_collection(
    molecules: List[Molecule],
    drivers: Union[DriverEnum, List[DriverEnum]],
    monkeypatch,
) -> BasicResultCollection:

    if not isinstance(drivers, list):
        drivers = [drivers]

    collection = BasicResultCollection(
        entries={
            DEFAULT_SERVER_ADDRESS: [
                BasicResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=molecule.to_smiles(mapped=True),
                    inchi_key=molecule.to_inchikey(),
                )
                for i, molecule in enumerate(molecules)
            ]
        }
    )

    def mock_return_result(molecule, driver):

        if driver == DriverEnum.gradient:
            return numpy.random.random((molecule.n_atoms, 3))
        elif driver == DriverEnum.hessian:
            return numpy.random.random((molecule.n_atoms * 3, molecule.n_atoms * 3))

        raise NotImplementedError()

    monkeypatch.setattr(
        BasicResultCollection,
        "to_records",
        lambda self: [
            (
                ResultRecord(
                    id=entry.record_id,
                    program="psi4",
                    driver=driver,
                    method="scf",
                    basis="sto-3g",
                    molecule=entry.record_id,
                    status=RecordStatusEnum.complete,
                    client=_FractalClient(address=address),
                    return_result=mock_return_result(
                        molecules[int(entry.record_id) - 1], driver
                    ),
                ),
                molecules[int(entry.record_id) - 1],
            )
            for address, entries in self.entries.items()
            for entry in entries
            for driver in drivers
        ],
    )

    return collection


def mock_optimization_result_collection(
    molecules: List[Molecule], monkeypatch
) -> OptimizationResultCollection:

    collection = OptimizationResultCollection(
        entries={
            DEFAULT_SERVER_ADDRESS: [
                OptimizationResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=molecule.to_smiles(mapped=True),
                    inchi_key=molecule.to_inchikey(),
                )
                for i, molecule in enumerate(molecules)
            ]
        }
    )

    monkeypatch.setattr(
        OptimizationResultCollection,
        "to_records",
        lambda self: [
            (
                OptimizationRecord(
                    id=entry.record_id,
                    program="psi4",
                    qc_spec=QCSpecification(
                        driver=DriverEnum.gradient,
                        method="scf",
                        basis="sto-3g",
                        program="psi4",
                    ),
                    initial_molecule=ObjectId(entry.record_id),
                    final_molecule=ObjectId(entry.record_id),
                    status=RecordStatusEnum.complete,
                    energies=[numpy.random.random()],
                    client=_FractalClient(address=address),
                ),
                molecules[int(entry.record_id) - 1],
            )
            for address, entries in self.entries.items()
            for entry in entries
        ],
    )

    monkeypatch.setattr(
        OptimizationResultCollection,
        "to_basic_result_collection",
        lambda self, driver: mock_basic_result_collection(
            molecules, driver, monkeypatch
        ),
    )

    return collection
