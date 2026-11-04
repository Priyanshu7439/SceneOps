
import pytest
from pydantic import ValidationError

from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    DeteriorationManager,
    PatientChanges,
)
from sceneops.simulator.patients import (
    InjuryType,
    MentalStatus,
    Patient,
)


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


def make_event(
    event_id="E001",
    patient_id="P001",
    at_seconds=60,
    changes=None,
):
    return DeteriorationEvent(
        event_id=event_id,
        patient_id=patient_id,
        at_seconds=at_seconds,
        changes=changes or PatientChanges(respiratory_rate=35),
    )


def test_event_is_not_applied_before_scheduled_time():
    patient = make_patient()
    manager = DeteriorationManager([make_event()])

    applied = manager.apply_due_events({"P001": patient}, 30)

    assert applied == []
    assert patient.respiratory_rate == 20


def test_event_is_applied_at_scheduled_time():
    patient = make_patient()
    manager = DeteriorationManager([make_event()])

    applied = manager.apply_due_events({"P001": patient}, 60)

    assert len(applied) == 1
    assert patient.respiratory_rate == 35


def test_due_event_is_applied_when_time_moves_past_schedule():
    patient = make_patient()
    manager = DeteriorationManager([make_event(at_seconds=60)])

    applied = manager.apply_due_events({"P001": patient}, 120)

    assert len(applied) == 1
    assert patient.respiratory_rate == 35


def test_event_is_applied_only_once():
    patient = make_patient()
    manager = DeteriorationManager([make_event()])

    first = manager.apply_due_events({"P001": patient}, 60)
    second = manager.apply_due_events({"P001": patient}, 90)

    assert len(first) == 1
    assert second == []
    assert manager.applied_event_ids == frozenset({"E001"})


def test_multiple_events_are_applied_in_timestamp_order():
    patient = make_patient()

    events = [
        make_event("E002", at_seconds=120,
                   changes=PatientChanges(respiratory_rate=40)),
        make_event("E001", at_seconds=60,
                   changes=PatientChanges(respiratory_rate=35)),
    ]

    manager = DeteriorationManager(events)
    applied = manager.apply_due_events({"P001": patient}, 120)

    assert [event.event_id for event in applied] == ["E001", "E002"]
    assert patient.respiratory_rate == 40


def test_multiple_patients_can_be_updated():
    p1 = make_patient("P001")
    p2 = make_patient("P002")

    events = [
        make_event("E001", "P001", changes=PatientChanges(
            respiratory_rate=35
        )),
        make_event("E002", "P002", changes=PatientChanges(
            follows_commands=False
        )),
    ]

    manager = DeteriorationManager(events)
    manager.apply_due_events({"P001": p1, "P002": p2}, 60)

    assert p1.respiratory_rate == 35
    assert p2.follows_commands is False


def test_multiple_fields_can_change_in_one_event():
    patient = make_patient()

    event = make_event(
        changes=PatientChanges(
            respiratory_rate=35,
            radial_pulse_present=False,
            mental_status=MentalStatus.CONFUSED,
        )
    )

    manager = DeteriorationManager([event])
    applied = manager.apply_due_events({"P001": patient}, 60)

    assert patient.respiratory_rate == 35
    assert patient.radial_pulse_present is False
    assert patient.mental_status == MentalStatus.CONFUSED
    assert len(applied[0].changed_fields) == 3


def test_event_for_unknown_patient_is_rejected():
    manager = DeteriorationManager([make_event()])

    with pytest.raises(KeyError):
        manager.apply_due_events({}, 60)


def test_unknown_patient_does_not_mark_event_as_applied():
    manager = DeteriorationManager([make_event()])

    with pytest.raises(KeyError):
        manager.apply_due_events({}, 60)

    assert manager.applied_event_ids == frozenset()


def test_duplicate_event_ids_are_rejected():
    with pytest.raises(ValueError):
        DeteriorationManager([
            make_event("E001"),
            make_event("E001", at_seconds=120),
        ])


def test_empty_patient_changes_are_rejected():
    with pytest.raises(ValidationError):
        PatientChanges()


def test_invalid_respiratory_rate_is_rejected():
    with pytest.raises(ValidationError):
        PatientChanges(respiratory_rate=100)


def test_simulation_time_cannot_move_backwards():
    manager = DeteriorationManager([])
    manager.apply_due_events({}, 120)

    with pytest.raises(ValueError):
        manager.apply_due_events({}, 60)


def test_boolean_timestamp_is_rejected():
    manager = DeteriorationManager([])

    with pytest.raises(TypeError):
        manager.apply_due_events({}, True)
