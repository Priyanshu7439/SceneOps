

from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.engine import SimulationStatus
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_builder import ScenarioBuilder


def create_scenario(
    deterioration_events: list[DeteriorationEvent] | None = None,
) -> ScenarioDefinition:
    patient = Patient(
        patient_id="P001",
        age=30,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=20,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
    )

    ambulance = Ambulance(ambulance_id="A001")

    return ScenarioDefinition(
        scenario_id="SCN001",
        title="Builder Test Scenario",
        description="Synthetic scenario for testing.",
        duration_seconds=600,
        patients=[patient],
        ambulances=[ambulance],
        deterioration_events=deterioration_events or [],
    )


def test_builder_creates_engine_with_registered_patients():
    scenario = create_scenario()

    engine = ScenarioBuilder.build(scenario)

    assert engine.status == SimulationStatus.CREATED
    assert "P001" in engine.patients
    assert engine.patients["P001"].respiratory_rate == 20


def test_builder_passes_deterioration_events_to_engine():
    event = DeteriorationEvent(
        event_id="EV001",
        patient_id="P001",
        at_seconds=60,
        changes=PatientChanges(respiratory_rate=35),
    )

    scenario = create_scenario([event])
    engine = ScenarioBuilder.build(scenario)

    engine.start()
    engine.advance_time(60)

    assert engine.patients["P001"].respiratory_rate == 35
