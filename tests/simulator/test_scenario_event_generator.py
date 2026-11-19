from sceneops.simulator.deterioration import PatientChanges
from sceneops.simulator.patient_profile_generator import (
    PatientProfileGenerator,
)
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
    ScenarioEventSpec,
)
from sceneops.simulator.scenario_event_generator import (
    ScenarioEventGenerator,
)


def create_config(seed=42):
    event = ScenarioEventSpec(
        event_id="EV001",
        target_archetype=PatientArchetype.RED,
        earliest_seconds=100,
        latest_seconds=200,
        changes=PatientChanges(respiratory_rate=40),
    )

    return ScenarioConfig(
        scenario_id="MCI-EVENT-001",
        seed=seed,
        event_specs=(event,),
    )


def test_generates_configured_events():
    config = create_config()
    patients = PatientProfileGenerator().generate(config)

    events = ScenarioEventGenerator().generate(config, patients)

    assert len(events) == 1
    assert events[0].event_id == "EV001"


def test_event_time_stays_within_configured_window():
    config = create_config()
    patients = PatientProfileGenerator().generate(config)

    events = ScenarioEventGenerator().generate(config, patients)

    assert 100 <= events[0].at_seconds <= 200


def test_event_targets_configured_archetype():
    config = create_config()
    patients = PatientProfileGenerator().generate(config)

    events = ScenarioEventGenerator().generate(config, patients)

    selected_patient = next(
        patient for patient in patients if patient.patient_id == events[0].patient_id
    )

    from sceneops.simulator.triage import (
        TriageCategory,
        classify_patient,
    )

    assert classify_patient(selected_patient).category == TriageCategory.RED


def test_same_seed_produces_same_events():
    config = create_config()
    generator = PatientProfileGenerator()
    patients = generator.generate(config)

    first = ScenarioEventGenerator().generate(config, patients)
    second = ScenarioEventGenerator().generate(config, patients)

    assert [event.model_dump() for event in first] == [
        event.model_dump() for event in second
    ]


def test_empty_event_configuration_returns_empty_list():
    config = ScenarioConfig(
        scenario_id="MCI-EVENT-002",
        seed=42,
    )
    patients = PatientProfileGenerator().generate(config)

    events = ScenarioEventGenerator().generate(config, patients)

    assert events == []


def test_generates_multiple_events():
    first_event = ScenarioEventSpec(
        event_id="EV001",
        target_archetype=PatientArchetype.RED,
        earliest_seconds=100,
        latest_seconds=200,
        changes=PatientChanges(respiratory_rate=40),
    )
    second_event = ScenarioEventSpec(
        event_id="EV002",
        target_archetype=PatientArchetype.YELLOW,
        earliest_seconds=200,
        latest_seconds=300,
        changes=PatientChanges(follows_commands=False),
    )

    config = ScenarioConfig(
        scenario_id="MCI-EVENT-003",
        seed=42,
        event_specs=(first_event, second_event),
    )
    patients = PatientProfileGenerator().generate(config)

    events = ScenarioEventGenerator().generate(config, patients)

    assert len(events) == 2
    assert {event.event_id for event in events} == {
        "EV001",
        "EV002",
    }
