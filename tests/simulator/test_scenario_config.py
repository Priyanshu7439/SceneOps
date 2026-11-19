import pytest
from pydantic import ValidationError

from sceneops.simulator.deterioration import PatientChanges
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
    ScenarioEventSpec,
)


def test_default_archetype_distribution():
    config = ScenarioConfig(
        scenario_id="MCI-NEW-001",
        seed=42,
    )

    assert config.patient_count == 20
    assert sum(config.archetype_counts.values()) == 20


def test_rejects_incorrect_archetype_distribution():
    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario_id="MCI-NEW-002",
            seed=42,
            patient_count=20,
            archetype_counts={
                PatientArchetype.GREEN: 10,
                PatientArchetype.YELLOW: 10,
            },
        )


def test_rejects_event_after_scenario_duration():
    event = ScenarioEventSpec(
        event_id="EV001",
        target_archetype=PatientArchetype.RED,
        earliest_seconds=500,
        latest_seconds=700,
        changes=PatientChanges(respiratory_rate=35),
    )

    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario_id="MCI-NEW-003",
            seed=42,
            event_specs=(event,),
        )


def test_rejects_duplicate_event_ids():
    event = ScenarioEventSpec(
        event_id="EV001",
        target_archetype=PatientArchetype.RED,
        earliest_seconds=100,
        latest_seconds=200,
        changes=PatientChanges(respiratory_rate=35),
    )

    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario_id="MCI-NEW-004",
            seed=42,
            event_specs=(event, event),
        )


def test_accepts_valid_event_configuration():
    event = ScenarioEventSpec(
        event_id="EV001",
        target_archetype=PatientArchetype.RED,
        earliest_seconds=100,
        latest_seconds=200,
        changes=PatientChanges(respiratory_rate=35),
    )

    config = ScenarioConfig(
        scenario_id="MCI-NEW-005",
        seed=42,
        event_specs=(event,),
    )

    assert config.event_specs[0].event_id == "EV001"
