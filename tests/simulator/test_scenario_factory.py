
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

    categories = {
        classify_patient(patient).category
        for patient in scenario.patients
    }

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
