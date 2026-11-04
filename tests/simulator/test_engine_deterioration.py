
import pytest

from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.engine import (
    SimulationEngine,
    SimulationStatus,
)
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.triage import TriageCategory


def make_patient(patient_id="P001"):
    return Patient(
        patient_id=patient_id,
        age=35,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=20,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
    )


def make_event():
    return DeteriorationEvent(
        event_id="E001",
        patient_id="P001",
        at_seconds=60,
        changes=PatientChanges(respiratory_rate=35),
    )


def make_engine(events=None):
    return SimulationEngine(
        ambulances=[Ambulance(ambulance_id="A001")],
        deterioration_events=events or [],
    )


def test_deterioration_event_is_applied_through_engine():
    engine = make_engine([make_event()])
    engine.add_patient(make_patient())
    engine.start()

    engine.advance_time(60)

    assert engine.patients["P001"].respiratory_rate == 35


def test_engine_reassesses_patient_after_state_change():
    engine = make_engine([make_event()])
    engine.add_patient(make_patient())
    engine.start()

    engine.assess_patient("P001")
    assert engine.triage_results["P001"].category == TriageCategory.YELLOW

    engine.advance_time(60)

    assert engine.triage_results["P001"].category == TriageCategory.RED


def test_engine_records_deterioration_and_reassessment_events():
    engine = make_engine([make_event()])
    engine.add_patient(make_patient())
    engine.start()

    engine.advance_time(60)

    event_types = [event.event_type for event in engine.events]

    assert "patient_state_changed" in event_types
    assert "patient_assessed" in event_types
    assert "patient_reassessed" in event_types


def test_scheduled_event_for_unregistered_patient_blocks_start():
    engine = make_engine([make_event()])

    with pytest.raises(ValueError):
        engine.start()

    assert engine.status == SimulationStatus.CREATED


def test_deterioration_event_is_not_applied_twice():
    engine = make_engine([make_event()])
    engine.add_patient(make_patient())
    engine.start()

    engine.advance_time(60)
    event_count = len([
        event for event in engine.events
        if event.event_type == "patient_state_changed"
    ])

    engine.advance_time(60)

    updated_event_count = len([
        event for event in engine.events
        if event.event_type == "patient_state_changed"
    ])

    assert event_count == 1
    assert updated_event_count == 1
