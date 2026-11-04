
import pytest

from sceneops.simulator.resources import (
    Ambulance,
    AmbulanceStatus,
    ResourceManager,
)
from sceneops.simulator.triage import TriageCategory, TriageResult


def make_result(patient_id, category):
    return TriageResult(
        patient_id=patient_id,
        category=category,
        reason="Test result",
        criteria=("test",),
    )


def make_manager(count=2):
    ambulances = [
        Ambulance(ambulance_id=f"A{i:03}")
        for i in range(1, count + 1)
    ]
    return ResourceManager(ambulances)


def test_all_ambulances_start_available():
    manager = make_manager()

    assert len(manager.available_ambulances()) == 2


def test_allocate_patient_to_available_ambulance():
    manager = make_manager()

    result = manager.allocate_patient("P001", TriageCategory.RED)

    assert result is not None
    assert result.patient_id == "P001"
    assert result.ambulance_id == "A001"
    assert manager.ambulances["A001"].status == AmbulanceStatus.OCCUPIED


def test_occupied_ambulance_is_not_reassigned():
    manager = make_manager(count=1)

    manager.allocate_patient("P001", TriageCategory.RED)
    result = manager.allocate_patient("P002", TriageCategory.YELLOW)

    assert result is None
    assert manager.ambulances["A001"].assigned_patient_id == "P001"


def test_allocation_returns_none_when_no_ambulance_is_available():
    manager = make_manager(count=0)

    result = manager.allocate_patient("P001", TriageCategory.RED)

    assert result is None


def test_black_patient_is_not_allocated():
    manager = make_manager()

    result = manager.allocate_patient("P001", TriageCategory.BLACK)

    assert result is None


def test_pending_patient_is_not_allocated():
    manager = make_manager()

    result = manager.allocate_patient("P001", TriageCategory.PENDING)

    assert result is None


def test_allocate_next_prioritizes_red_patient():
    manager = make_manager(count=1)

    waiting = [
        make_result("P001", TriageCategory.GREEN),
        make_result("P002", TriageCategory.YELLOW),
        make_result("P003", TriageCategory.RED),
    ]

    result = manager.allocate_next(waiting)

    assert result is not None
    assert result.patient_id == "P003"
    assert result.triage_category == TriageCategory.RED


def test_equal_priority_preserves_arrival_order():
    manager = make_manager(count=1)

    waiting = [
        make_result("P001", TriageCategory.RED),
        make_result("P002", TriageCategory.RED),
    ]

    result = manager.allocate_next(waiting)

    assert result is not None
    assert result.patient_id == "P001"


def test_release_ambulance():
    manager = make_manager(count=1)

    manager.allocate_patient("P001", TriageCategory.RED)
    released_patient = manager.release_ambulance("A001")

    assert released_patient == "P001"
    assert manager.ambulances["A001"].status == AmbulanceStatus.AVAILABLE


def test_cannot_release_already_available_ambulance():
    manager = make_manager(count=1)

    with pytest.raises(ValueError):
        manager.release_ambulance("A001")


def test_unknown_ambulance_raises_error():
    manager = make_manager()

    with pytest.raises(KeyError):
        manager.release_ambulance("A999")


def test_same_patient_cannot_be_assigned_twice():
    manager = make_manager(count=2)

    manager.allocate_patient("P001", TriageCategory.RED)

    with pytest.raises(ValueError):
        manager.allocate_patient("P001", TriageCategory.YELLOW)


def test_duplicate_ambulance_ids_are_rejected():
    with pytest.raises(ValueError):
        ResourceManager([
            Ambulance(ambulance_id="A001"),
            Ambulance(ambulance_id="A001"),
        ])
