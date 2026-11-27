from sceneops.simulator.engine import SimulationStatus
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.scenario_factory import (
    create_mass_casualty_scenario,
)
from sceneops.simulator.triage import TriageCategory, classify_patient


def test_factory_creates_20_patients():
    scenario = create_mass_casualty_scenario()

    assert len(scenario.patients) == 20
    assert len({p.patient_id for p in scenario.patients}) == 20


def test_factory_creates_five_ambulances():
    scenario = create_mass_casualty_scenario()

    assert len(scenario.ambulances) == 5


def test_factory_covers_all_triage_categories():
    scenario = create_mass_casualty_scenario()

    categories = {classify_patient(patient).category for patient in scenario.patients}

    assert categories == {
        TriageCategory.GREEN,
        TriageCategory.YELLOW,
        TriageCategory.RED,
        TriageCategory.BLACK,
        TriageCategory.PENDING,
    }


def test_factory_scenario_runs_deterioration_events():
    scenario = create_mass_casualty_scenario()
    engine = ScenarioBuilder.build(scenario)

    assert engine.status == SimulationStatus.CREATED

    engine.start()
    engine.advance_time(120)

    assert engine.patients["P005"].respiratory_rate == 35

    engine.advance_time(120)

    assert engine.patients["P006"].follows_commands is False


def test_factory_preserves_original_scenario_metadata():
    scenario = create_mass_casualty_scenario(1)

    assert scenario.scenario_id == "MCI-001"
    assert scenario.title == "20-Patient Mass Casualty Exercise"
    assert scenario.description == (
        "Synthetic educational scenario with 20 patients, "
        "five ambulances and three timed deterioration events."
    )
    assert scenario.duration_seconds == 600


def test_factory_preserves_original_deterioration_timings():
    scenario = create_mass_casualty_scenario(1)

    events = {event.event_id: event for event in scenario.deterioration_events}

    assert events["EV001"].patient_id == "P005"
    assert events["EV001"].at_seconds == 120

    assert events["EV002"].patient_id == "P006"
    assert events["EV002"].at_seconds == 240


def test_factory_generates_indexed_scenario_variations():
    scenario_1 = create_mass_casualty_scenario(1)
    scenario_2 = create_mass_casualty_scenario(2)

    assert scenario_1.scenario_id == "MCI-001"
    assert scenario_2.scenario_id == "MCI-002"

    assert scenario_1.title != scenario_2.title
    assert scenario_1.description != scenario_2.description

    assert (
        scenario_1.deterioration_events[0].at_seconds
        != scenario_2.deterioration_events[0].at_seconds
    )
