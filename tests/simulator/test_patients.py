
import pytest
from pydantic import ValidationError

from sceneops.simulator.patients import (
    InjuryType,
    MentalStatus,
    Patient,
)


def create_valid_patient() -> Patient:
    return Patient(
        patient_id="P001",
        age=35,
        injury_type=InjuryType.TRAUMA,
        mental_status=MentalStatus.ALERT,
        ambulatory=True,
        respiratory_rate=18,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
        personality="anxious",
        speech_style="short sentences",
        pain_level=4,
    )


def test_patient_creation():
    patient = create_valid_patient()

    assert patient.patient_id == "P001"
    assert patient.age == 35
    assert patient.injury_type == InjuryType.TRAUMA
    assert patient.pain_level == 4



def test_patient_age_validation():
    with pytest.raises(ValidationError):
        Patient(
            patient_id="P003",
            age=150,
            injury_type=InjuryType.TRAUMA,
            ambulatory=True,
            radial_pulse_present=True,
            follows_commands=True,
            major_hemorrhage=False,
        )



def test_patient_pain_update():
    patient = create_valid_patient()

    patient.update_pain(8)

    assert patient.pain_level == 8


def test_invalid_pain_assignment():
    patient = create_valid_patient()

    with pytest.raises(ValidationError):
        patient.pain_level = 15


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Patient(
            patient_id="P002",
            age=25,
            injury_type=InjuryType.BURN,
            ambulatory=False,
            radial_pulse_present=True,
            follows_commands=True,
            major_hemorrhage=False,
            unknown_field="unexpected",
        )
