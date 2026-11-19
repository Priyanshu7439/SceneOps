import pytest

from sceneops.agents.pain_state_tracker import PainStateTracker
from sceneops.simulator.patients import InjuryType, Patient


def make_patient(
    patient_id: str = "P001",
    pain_level: int = 6,
) -> Patient:
    return Patient(
        patient_id=patient_id,
        age=32,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=24,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
        personality="anxious",
        speech_style="plain",
        pain_level=pain_level,
    )


def test_first_observation_establishes_baseline():
    tracker = PainStateTracker()

    transition = tracker.observe(make_patient(pain_level=6))

    assert transition.previous.value == 6
    assert transition.current.value == 6
    assert transition.changed is False
    assert transition.direction == "unchanged"


def test_detects_pain_increase():
    tracker = PainStateTracker()

    tracker.observe(make_patient(pain_level=6))
    transition = tracker.observe(make_patient(pain_level=8))

    assert transition.previous.value == 6
    assert transition.current.value == 8
    assert transition.changed is True
    assert transition.direction == "increase"


def test_detects_pain_decrease():
    tracker = PainStateTracker()

    tracker.observe(make_patient(pain_level=8))
    transition = tracker.observe(make_patient(pain_level=4))

    assert transition.previous.value == 8
    assert transition.current.value == 4
    assert transition.changed is True
    assert transition.direction == "decrease"


def test_tracks_patients_independently():
    tracker = PainStateTracker()

    tracker.observe(make_patient("P001", 6))
    tracker.observe(make_patient("P002", 3))

    transition = tracker.observe(make_patient("P001", 8))

    assert transition.previous.value == 6
    assert transition.current.value == 8


def test_clear_patient_resets_baseline():
    tracker = PainStateTracker()

    tracker.observe(make_patient(pain_level=6))
    tracker.observe(make_patient(pain_level=8))

    tracker.clear_patient("P001")

    transition = tracker.observe(make_patient(pain_level=4))

    assert transition.previous.value == 4
    assert transition.current.value == 4
    assert transition.direction == "unchanged"


def test_clear_all_resets_every_patient():
    tracker = PainStateTracker()

    tracker.observe(make_patient("P001", 6))
    tracker.observe(make_patient("P002", 3))

    tracker.clear_all()

    first = tracker.observe(make_patient("P001", 8))
    second = tracker.observe(make_patient("P002", 5))

    assert first.previous.value == 8
    assert second.previous.value == 5


def test_rejects_non_patient_input():
    tracker = PainStateTracker()

    with pytest.raises(TypeError, match="patient must be a Patient"):
        tracker.observe({"patient_id": "P001", "pain_level": 6})
