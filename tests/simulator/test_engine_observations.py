
import pytest

from sceneops.simulator.clock import SimulationClock
from sceneops.simulator.engine import SimulationEngine
from sceneops.simulator.observations import ObservationType
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance


def make_patient() -> Patient:
    return Patient(
        patient_id="P001",
        age=35,
        injury_type=InjuryType.RESPIRATORY,
        ambulatory=False,
        respiratory_rate=37,
        radial_pulse_present=False,
        follows_commands=False,
        major_hemorrhage=True,
        capillary_refill_seconds=3.5,
        pain_level=8,
    )


def make_engine() -> SimulationEngine:
    engine = SimulationEngine(
        ambulances=[
            Ambulance(ambulance_id="A001"),
        ],
        clock=SimulationClock(),
    )
    engine.add_patient(make_patient())
    return engine


def test_engine_records_respiratory_rate_observation():
    engine = make_engine()

    observation = engine.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    assert observation.value == "37"
    assert observation.observed_at_seconds == 0
    assert engine.observation_registry.get_for_patient("P001") == [
        observation
    ]


def test_engine_records_radial_pulse_observation():
    engine = make_engine()

    observation = engine.observe_patient(
        "P001",
        ObservationType.RADIAL_PULSE,
    )

    assert observation.value == "absent"


def test_pain_observation_requires_dialogue_workflow():
    engine = make_engine()

    with pytest.raises(ValueError, match="dialogue or visual-observation workflow"):
        engine.observe_patient(
            "P001",
            ObservationType.REPORTED_PAIN,
        )


def test_observation_uses_current_simulation_time():
    engine = make_engine()
    engine.clock.advance(120)

    observation = engine.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    assert observation.observed_at_seconds == 120


def test_unknown_patient_is_rejected():
    engine = make_engine()

    with pytest.raises(KeyError, match="Unknown patient"):
        engine.observe_patient(
            "UNKNOWN",
            ObservationType.RESPIRATORY_RATE,
        )


def test_missing_measurement_is_rejected():
    engine = make_engine()
    engine.patients["P001"].respiratory_rate = None

    with pytest.raises(ValueError, match="no recorded value"):
        engine.observe_patient(
            "P001",
            ObservationType.RESPIRATORY_RATE,
        )


def test_unsupported_observation_requires_dialogue_workflow():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="dialogue or visual-observation workflow",
    ):
        engine.observe_patient(
            "P001",
            ObservationType.REPORTED_SYMPTOM,
        )


def test_observation_is_rejected_after_simulation_completion():
    engine = make_engine()
    engine.start()
    engine.complete()

    with pytest.raises(RuntimeError, match="unavailable"):
        engine.observe_patient(
            "P001",
            ObservationType.RESPIRATORY_RATE,
        )


def test_observation_records_simulation_event():
    engine = make_engine()

    engine.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    assert engine.events[-1].event_type == "learner_observation_recorded"
    assert engine.events[-1].patient_id == "P001"
