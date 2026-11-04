
import pytest

from sceneops.simulator.engine import (
    SimulationEngine,
    SimulationStatus,
)
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.triage import TriageCategory


def make_patient(patient_id="P001", ambulatory=False, **overrides):
    data = {
        "patient_id": patient_id,
        "age": 35,
        "injury_type": InjuryType.TRAUMA,
        "ambulatory": ambulatory,
        "respiratory_rate": 20,
        "radial_pulse_present": True,
        "follows_commands": True,
        "major_hemorrhage": False,
    }
    data.update(overrides)
    return Patient(**data)


def make_engine(ambulance_count=2):
    ambulances = [
        Ambulance(ambulance_id=f"A{i:03}")
        for i in range(1, ambulance_count + 1)
    ]
    return SimulationEngine(ambulances)


def test_engine_starts_in_created_state():
    engine = make_engine()

    assert engine.status == SimulationStatus.CREATED


def test_patient_registration():
    engine = make_engine()
    patient = make_patient()

    engine.add_patient(patient)

    assert "P001" in engine.patients
    assert engine.get_snapshot()["patient_count"] == 1


def test_duplicate_patient_registration_is_rejected():
    engine = make_engine()
    patient = make_patient()

    engine.add_patient(patient)

    with pytest.raises(ValueError):
        engine.add_patient(patient)


def test_patient_can_only_be_registered_before_start():
    engine = make_engine()
    engine.start()

    with pytest.raises(RuntimeError):
        engine.add_patient(make_patient())


def test_simulation_lifecycle():
    engine = make_engine()

    engine.start()
    assert engine.status == SimulationStatus.RUNNING

    engine.pause()
    assert engine.status == SimulationStatus.PAUSED

    engine.resume()
    assert engine.status == SimulationStatus.RUNNING

    engine.complete()
    assert engine.status == SimulationStatus.COMPLETED


def test_invalid_lifecycle_transition_is_rejected():
    engine = make_engine()

    with pytest.raises(RuntimeError):
        engine.pause()


def test_unknown_patient_assessment_is_rejected():
    engine = make_engine()
    engine.start()

    with pytest.raises(KeyError):
        engine.assess_patient("UNKNOWN")


def test_patient_assessment():
    engine = make_engine()
    engine.add_patient(make_patient(ambulatory=True))

    result = engine.assess_patient("P001")

    assert result.category == TriageCategory.GREEN
    assert "P001" in engine.triage_results


def test_clock_advances_only_while_running():
    engine = make_engine()

    with pytest.raises(RuntimeError):
        engine.advance_time(60)

    engine.start()
    snapshot = engine.advance_time(60)

    assert snapshot.elapsed_seconds == 60


def test_allocation_requires_running_simulation():
    engine = make_engine()

    with pytest.raises(RuntimeError):
        engine.allocate_next_patient()


def test_allocation_uses_triage_priority():
    engine = make_engine(ambulance_count=1)

    engine.add_patient(make_patient("P001", ambulatory=True))
    engine.add_patient(make_patient("P002"))

    engine.assess_patient("P001")
    engine.assess_patient("P002")

    engine.start()
    allocation = engine.allocate_next_patient()

    assert allocation is not None
    assert allocation.patient_id == "P002"
    assert allocation.triage_category == TriageCategory.YELLOW


def test_same_patient_is_not_allocated_twice():
    engine = make_engine(ambulance_count=2)
    engine.add_patient(make_patient("P001"))
    engine.assess_patient("P001")
    engine.start()

    first = engine.allocate_next_patient()
    second = engine.allocate_next_patient()

    assert first is not None
    assert first.patient_id == "P001"
    assert second is None


def test_ambulance_release_updates_allocation():
    engine = make_engine(ambulance_count=1)
    engine.add_patient(make_patient("P001"))
    engine.assess_patient("P001")
    engine.start()

    engine.allocate_next_patient()
    released_patient = engine.release_ambulance("A001")

    assert released_patient == "P001"
    assert "P001" not in engine.allocations


def test_events_are_recorded():
    engine = make_engine()
    engine.add_patient(make_patient())
    engine.start()

    event_types = [event.event_type for event in engine.events]

    assert "patient_registered" in event_types
    assert "simulation_started" in event_types


def test_completed_simulation_cannot_advance_time():
    engine = make_engine()
    engine.start()
    engine.complete()

    with pytest.raises(RuntimeError):
        engine.advance_time(60)
