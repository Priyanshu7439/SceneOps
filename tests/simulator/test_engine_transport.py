
import pytest

from sceneops.simulator.engine import (
    SimulationEngine,
    SimulationStatus,
)
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.transport import TransportStatus
from sceneops.simulator.triage import TriageCategory


def make_patient(patient_id: str) -> Patient:
    return Patient(
        patient_id=patient_id,
        age=35,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=35,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
    )


def create_running_engine(patient_count=1):
    ambulances = [
        Ambulance(ambulance_id=f"A{i:03}")
        for i in range(1, patient_count + 1)
    ]

    engine = SimulationEngine(ambulances)

    for i in range(1, patient_count + 1):
        patient = make_patient(f"P{i:03}")
        engine.add_patient(patient)
        result = engine.assess_patient(patient.patient_id)
        assert result.category == TriageCategory.RED

    engine.start()
    return engine


def test_engine_starts_transport_for_allocated_patient():
    engine = create_running_engine()

    allocation = engine.allocate_next_patient()
    assert allocation is not None

    record = engine.start_transport(
        patient_id=allocation.patient_id,
        travel_duration_seconds=120,
    )

    assert record.status == TransportStatus.IN_PROGRESS
    assert record.patient_id == allocation.patient_id
    assert record.ambulance_id == allocation.ambulance_id

    event_types = [event.event_type for event in engine.events]
    assert "transport_started" in event_types


def test_engine_does_not_complete_transport_too_early():
    engine = create_running_engine()

    allocation = engine.allocate_next_patient()
    assert allocation is not None

    engine.start_transport(allocation.patient_id, 120)

    engine.advance_time(119)

    record = next(iter(engine.transport_manager.transports.values()))

    assert record.status == TransportStatus.IN_PROGRESS
    assert allocation.patient_id not in engine.transported_patients
    assert allocation.patient_id in engine.allocations
    assert allocation.ambulance_id not in [
        ambulance.ambulance_id
        for ambulance in engine.resource_manager.available_ambulances()
    ]


def test_engine_completes_transport_at_correct_time():
    engine = create_running_engine()

    allocation = engine.allocate_next_patient()
    assert allocation is not None

    engine.start_transport(allocation.patient_id, 120)

    engine.advance_time(60)
    engine.advance_time(59)

    assert allocation.patient_id not in engine.transported_patients

    engine.advance_time(1)

    record = next(iter(engine.transport_manager.transports.values()))

    assert record.status == TransportStatus.COMPLETED
    assert record.completed_at_seconds == 120
    assert allocation.patient_id in engine.transported_patients
    assert allocation.patient_id not in engine.allocations
    assert allocation.ambulance_id in [
        ambulance.ambulance_id
        for ambulance in engine.resource_manager.available_ambulances()
    ]

    event_types = [event.event_type for event in engine.events]
    assert "transport_completed" in event_types


def test_engine_reuses_ambulance_after_transport_completion():
    engine = create_running_engine(patient_count=2)

    first = engine.allocate_next_patient()
    assert first is not None

    engine.start_transport(first.patient_id, 120)
    engine.advance_time(120)

    second = engine.allocate_next_patient()

    assert second is not None
    assert second.patient_id != first.patient_id
    assert second.ambulance_id == first.ambulance_id


def test_engine_blocks_manual_release_during_active_transport():
    engine = create_running_engine()

    allocation = engine.allocate_next_patient()
    assert allocation is not None

    engine.start_transport(allocation.patient_id, 120)

    with pytest.raises(ValueError, match="during transport"):
        engine.release_ambulance(allocation.ambulance_id)


def test_engine_rejects_transport_for_unallocated_patient():
    engine = create_running_engine()

    with pytest.raises(ValueError, match="no active ambulance allocation"):
        engine.start_transport("P001", 120)
