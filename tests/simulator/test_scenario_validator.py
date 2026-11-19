import pytest

from sceneops.simulator.deterioration import PatientChanges
from sceneops.simulator.patient_profile_generator import (
    PatientProfileGenerator,
)
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
    ScenarioEventSpec,
)
from sceneops.simulator.scenario_event_generator import (
    ScenarioEventGenerator,
)
from sceneops.simulator.scenario_validator import ScenarioValidator


def create_config():
    event = ScenarioEventSpec(
        event_id="EV001",
        target_archetype=PatientArchetype.RED,
        earliest_seconds=100,
        latest_seconds=200,
        changes=PatientChanges(respiratory_rate=40),
    )

    return ScenarioConfig(
        scenario_id="MCI-VALIDATOR-001",
        seed=42,
        event_specs=(event,),
    )


def create_valid_scenario(config):
    patients = PatientProfileGenerator().generate(config)

    events = ScenarioEventGenerator().generate(
        config,
        patients,
    )

    ambulances = [
        Ambulance(ambulance_id=f"A{index:03d}")
        for index in range(1, config.ambulance_count + 1)
    ]

    return ScenarioDefinition(
        scenario_id=config.scenario_id,
        title="Validator Test Scenario",
        description="Synthetic validator test scenario.",
        duration_seconds=config.duration_seconds,
        patients=patients,
        ambulances=ambulances,
        deterioration_events=events,
    )


def test_accepts_valid_scenario():
    config = create_config()
    scenario = create_valid_scenario(config)

    result = ScenarioValidator().validate(config, scenario)

    assert result is None


def test_rejects_mismatched_scenario_id():
    config = create_config()
    scenario = create_valid_scenario(config)
    scenario.scenario_id = "MCI-WRONG-001"

    with pytest.raises(ValueError, match="Scenario ID"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_mismatched_duration():
    config = create_config()
    scenario = create_valid_scenario(config)
    scenario.duration_seconds = 500

    with pytest.raises(ValueError, match="duration"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_incorrect_patient_count():
    config = create_config()
    scenario = create_valid_scenario(config)
    scenario.patients.pop()

    with pytest.raises(ValueError, match="patient count"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_incorrect_ambulance_count():
    config = create_config()
    scenario = create_valid_scenario(config)
    scenario.ambulances.pop()

    with pytest.raises(ValueError, match="ambulance count"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_duplicate_patient_ids():
    config = create_config()
    scenario = create_valid_scenario(config)
    scenario.patients[1].patient_id = scenario.patients[0].patient_id

    with pytest.raises(ValueError, match="Patient IDs"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_incorrect_archetype_distribution():
    config = create_config()
    scenario = create_valid_scenario(config)

    green_patient = next(patient for patient in scenario.patients if patient.ambulatory)
    green_patient.ambulatory = False
    green_patient.respiratory_rate = 35

    with pytest.raises(ValueError, match="triage distribution"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_event_outside_configured_time_window():
    config = create_config()
    scenario = create_valid_scenario(config)

    original_event = scenario.deterioration_events[0]
    scenario.deterioration_events[0] = original_event.model_copy(
        update={"at_seconds": 250}
    )

    with pytest.raises(ValueError, match="time window"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_mismatched_event_ids():
    config = create_config()
    scenario = create_valid_scenario(config)

    original_event = scenario.deterioration_events[0]
    scenario.deterioration_events[0] = original_event.model_copy(
        update={"event_id": "EV999"}
    )

    with pytest.raises(ValueError, match="event IDs"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_event_changes_not_matching_configuration():
    config = create_config()
    scenario = create_valid_scenario(config)

    original_event = scenario.deterioration_events[0]
    scenario.deterioration_events[0] = original_event.model_copy(
        update={"changes": PatientChanges(respiratory_rate=45)}
    )

    with pytest.raises(ValueError, match="changes"):
        ScenarioValidator().validate(config, scenario)


def test_rejects_event_targeting_wrong_archetype():
    config = create_config()
    scenario = create_valid_scenario(config)

    green_patient = next(patient for patient in scenario.patients if patient.ambulatory)

    original_event = scenario.deterioration_events[0]
    scenario.deterioration_events[0] = original_event.model_copy(
        update={"patient_id": green_patient.patient_id}
    )

    with pytest.raises(ValueError, match="outside"):
        ScenarioValidator().validate(config, scenario)
