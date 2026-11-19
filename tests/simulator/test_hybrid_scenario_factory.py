import pytest

from sceneops.simulator.deterioration import PatientChanges
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
    ScenarioEventSpec,
)
from sceneops.simulator.scenario_factory import (
    create_hybrid_scenario,
    create_mass_casualty_scenario,
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
        scenario_id="MCI-HYBRID-001",
        seed=seed,
        event_specs=(event,),
    )


def test_generates_valid_hybrid_scenario():
    config = create_config()

    scenario = create_hybrid_scenario(config)

    assert scenario.scenario_id == config.scenario_id
    assert len(scenario.patients) == config.patient_count
    assert len(scenario.ambulances) == config.ambulance_count
    assert len(scenario.deterioration_events) == 1


def test_same_seed_produces_same_scenario():
    config = create_config()

    first = create_hybrid_scenario(config)
    second = create_hybrid_scenario(config)

    assert first.model_dump() == second.model_dump()


def test_different_seeds_produce_different_scenarios():
    first = create_hybrid_scenario(create_config(seed=42))
    second = create_hybrid_scenario(create_config(seed=99))

    assert first.model_dump() != second.model_dump()


def test_generates_scenario_without_events():
    config = ScenarioConfig(
        scenario_id="MCI-HYBRID-002",
        seed=42,
    )

    scenario = create_hybrid_scenario(config)

    assert scenario.deterioration_events == []


def test_rejects_invalid_config_type():
    with pytest.raises(TypeError, match="ScenarioConfig"):
        create_hybrid_scenario("invalid")


def test_existing_factory_remains_unchanged():
    scenario = create_mass_casualty_scenario(1)

    assert scenario.scenario_id == "MCI-001"
    assert len(scenario.patients) == 20
    assert len(scenario.ambulances) == 5
    assert len(scenario.deterioration_events) == 2
