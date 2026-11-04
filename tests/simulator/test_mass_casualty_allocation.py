
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.scenario_factory import (
    create_mass_casualty_scenario,
)
from sceneops.simulator.triage import TriageCategory


def create_assessed_engine():
    scenario = create_mass_casualty_scenario()
    engine = ScenarioBuilder.build(scenario)

    for patient in scenario.patients:
        engine.assess_patient(patient.patient_id)

    engine.start()
    return engine


def test_only_five_patients_receive_ambulances():
    engine = create_assessed_engine()

    allocations = [
        engine.allocate_next_patient()
        for _ in range(5)
    ]

    successful_allocations = [
        allocation
        for allocation in allocations
        if allocation is not None
    ]

    assert len(successful_allocations) == 5
    assert len(engine.allocations) == 5


def test_red_patients_receive_priority_over_yellow_and_green():
    engine = create_assessed_engine()

    first_allocation = engine.allocate_next_patient()

    assert first_allocation is not None
    assert first_allocation.patient_id == "P009"
    assert engine.triage_results["P009"].category == TriageCategory.RED


def test_black_and_pending_patients_are_not_allocated():
    engine = create_assessed_engine()

    allocated_patient_ids = set()

    for _ in range(5):
        result = engine.allocate_next_patient()
        if result is not None:
            allocated_patient_ids.add(result.patient_id)

    assert not allocated_patient_ids.intersection(
        {"P013", "P014", "P015", "P016",
         "P017", "P018", "P019", "P020"}
    )


def test_released_ambulance_can_be_reallocated():
    engine = create_assessed_engine()

    first_allocation = engine.allocate_next_patient()

    assert first_allocation is not None

    engine.release_ambulance(first_allocation.ambulance_id)

    next_allocation = engine.allocate_next_patient()

    assert next_allocation is not None
    assert next_allocation.patient_id == first_allocation.patient_id
    assert first_allocation.patient_id not in engine.transported_patients