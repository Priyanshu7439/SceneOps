


from sceneops.simulator.patients import (
    InjuryType,
    Patient,
)
from sceneops.simulator.triage import (
    TriageCategory,
    classify_patient,
)


def make_patient(**overrides) -> Patient:
    data = {
        "patient_id": "P001",
        "age": 35,
        "injury_type": InjuryType.TRAUMA,
        "ambulatory": False,
        "respiratory_rate": 20,
        "radial_pulse_present": True,
        "follows_commands": True,
        "major_hemorrhage": False,
        "airway_opened": False,
    }
    data.update(overrides)
    return Patient(**data)


def test_ambulatory_patient_is_green():
    patient = make_patient(ambulatory=True)

    result = classify_patient(patient)

    assert result.category == TriageCategory.GREEN


def test_respiratory_rate_over_30_is_red():
    patient = make_patient(respiratory_rate=31)

    result = classify_patient(patient)

    assert result.category == TriageCategory.RED


def test_absent_radial_pulse_is_red():
    patient = make_patient(radial_pulse_present=False)

    result = classify_patient(patient)

    assert result.category == TriageCategory.RED


def test_delayed_capillary_refill_is_red():
    patient = make_patient(capillary_refill_seconds=3)

    result = classify_patient(patient)

    assert result.category == TriageCategory.RED


def test_patient_unable_to_follow_commands_is_red():
    patient = make_patient(follows_commands=False)

    result = classify_patient(patient)

    assert result.category == TriageCategory.RED


def test_patient_passing_all_checks_is_yellow():
    patient = make_patient()

    result = classify_patient(patient)

    assert result.category == TriageCategory.YELLOW


def test_apneic_patient_requires_airway_assessment():
    patient = make_patient(respiratory_rate=0)

    result = classify_patient(patient)

    assert result.category == TriageCategory.PENDING


def test_patient_still_apneic_after_airway_repositioning_is_black():
    patient = make_patient(
        respiratory_rate=0,
        airway_opened=True,
    )

    result = classify_patient(patient)

    assert result.category == TriageCategory.BLACK


def test_missing_respiratory_assessment_is_pending():
    patient = make_patient(respiratory_rate=None)

    result = classify_patient(patient)

    assert result.category == TriageCategory.PENDING
