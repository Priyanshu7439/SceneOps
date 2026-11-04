import pytest
from pydantic import ValidationError

from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.scenario import ScenarioDefinition


def create_patient(patient_id: str = "P001") -> Patient:
    return Patient(
        patient_id=patient_id,
        age=30,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=20,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
    )


def create_ambulance(ambulance_id: str = "A001") -> Ambulance:
    return Ambulance(ambulance_id=ambulance_id)


def create_scenario(**overrides) -> ScenarioDefinition:
    data = {
        "scenario_id": "SCN001",
        "title": "Mass Casualty Exercise",
        "description": "Synthetic training scenario.",
        "duration_seconds": 600,
        "patients": [create_patient()],
        "ambulances": [create_ambulance()],
        "deterioration_events": [],
    }
    data.update(overrides)
    return ScenarioDefinition(**data)


def test_valid_scenario_creation():
    scenario = create_scenario()

    assert scenario.scenario_id == "SCN001"
    assert len(scenario.patients) == 1
    assert len(scenario.ambulances) == 1


def test_duplicate_patient_ids_are_rejected():
    with pytest.raises(ValidationError):
        create_scenario(
            patients=[
                create_patient("P001"),
                create_patient("P001"),
            ]
        )


def test_duplicate_ambulance_ids_are_rejected():
    with pytest.raises(ValidationError):
        create_scenario(
            ambulances=[
                create_ambulance("A001"),
                create_ambulance("A001"),
            ]
        )


def test_unknown_patient_event_is_rejected():
    event = DeteriorationEvent(
        event_id="EV001",
        patient_id="P999",
        at_seconds=60,
        changes=PatientChanges(respiratory_rate=35),
    )

    with pytest.raises(ValidationError):
        create_scenario(deterioration_events=[event])


def test_event_after_scenario_duration_is_rejected():
    event = DeteriorationEvent(
        event_id="EV002",
        patient_id="P001",
        at_seconds=700,
        changes=PatientChanges(respiratory_rate=35),
    )

    with pytest.raises(ValidationError):
        create_scenario(deterioration_events=[event])


def test_scenario_requires_at_least_one_patient():
    with pytest.raises(ValidationError):
        create_scenario(patients=[])


def test_scenario_requires_at_least_one_ambulance():
    with pytest.raises(ValidationError):
        create_scenario(ambulances=[])
