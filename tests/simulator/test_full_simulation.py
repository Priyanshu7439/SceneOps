
from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.engine import SimulationEngine
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.triage import TriageCategory


def test_complete_simulation_workflow():
    patient = Patient(
        patient_id="P001",
        age=35,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=20,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
    )

    event = DeteriorationEvent(
        event_id="E001",
        patient_id="P001",
        at_seconds=60,
        changes=PatientChanges(respiratory_rate=35),
    )

    engine = SimulationEngine(
        ambulances=[
            Ambulance(ambulance_id="A001"),
            Ambulance(ambulance_id="A002"),
        ],
        deterioration_events=[event],
    )

    # Register patient and perform initial assessment.
    engine.add_patient(patient)

    initial_result = engine.assess_patient("P001")
    assert initial_result.category == TriageCategory.YELLOW

    # Start simulation and advance time.
    engine.start()
    engine.advance_time(60)

    # Verify deterioration and automatic reassessment.
    assert patient.respiratory_rate == 35
    assert engine.triage_results["P001"].category == TriageCategory.RED

    # Allocate ambulance using updated triage result.
    allocation = engine.allocate_next_patient()

    assert allocation is not None
    assert allocation.patient_id == "P001"
    assert allocation.triage_category == TriageCategory.RED

    # Verify simulation state.
    snapshot = engine.get_snapshot()

    assert snapshot["status"] == "running"
    assert snapshot["clock"]["elapsed_seconds"] == 60
    assert snapshot["allocated_patient_count"] == 1

    # Complete simulation.
    engine.complete()

    assert engine.get_snapshot()["status"] == "completed"
