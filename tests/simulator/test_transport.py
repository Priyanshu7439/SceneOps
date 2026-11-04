
import pytest

from sceneops.simulator.resources import (
    Ambulance,
    ResourceManager,
)
from sceneops.simulator.transport import (
    TransportManager,
    TransportStatus,
)
from sceneops.simulator.triage import TriageCategory


def create_transport_setup():
    resource_manager = ResourceManager([
        Ambulance(ambulance_id="A001"),
    ])

    allocation = resource_manager.allocate_patient(
        patient_id="P001",
        category=TriageCategory.RED,
    )

    assert allocation is not None

    manager = TransportManager(resource_manager)
    return resource_manager, allocation, manager


def test_transport_starts_with_occupied_ambulance():
    resource_manager, allocation, manager = create_transport_setup()

    record = manager.start_transport(allocation, 0, 120)

    assert record.status == TransportStatus.IN_PROGRESS
    assert resource_manager.ambulances["A001"].assigned_patient_id == "P001"


def test_transport_completes_after_travel_duration():
    resource_manager, allocation, manager = create_transport_setup()

    manager.start_transport(allocation, 0, 120)

    completed = manager.advance_time(120)

    assert len(completed) == 1
    assert completed[0].status == TransportStatus.COMPLETED
    assert completed[0].completed_at_seconds == 120
    assert resource_manager.ambulances["A001"].assigned_patient_id is None


def test_transport_does_not_complete_too_early():
    _, allocation, manager = create_transport_setup()

    manager.start_transport(allocation, 0, 120)

    completed = manager.advance_time(60)

    assert completed == []


def test_completed_transport_is_not_completed_twice():
    _, allocation, manager = create_transport_setup()

    manager.start_transport(allocation, 0, 120)

    first = manager.advance_time(120)
    second = manager.advance_time(180)

    assert len(first) == 1
    assert second == []


def test_transport_rejects_duplicate_active_patient():
    _, allocation, manager = create_transport_setup()

    manager.start_transport(allocation, 0, 120)

    with pytest.raises(ValueError):
        manager.start_transport(allocation, 10, 60)


def test_transport_rejects_invalid_duration():
    _, allocation, manager = create_transport_setup()

    with pytest.raises(ValueError):
        manager.start_transport(allocation, 0, 0)


def test_released_ambulance_can_be_reused():
    resource_manager, allocation, manager = create_transport_setup()

    manager.start_transport(allocation, 0, 120)
    manager.advance_time(120)

    next_allocation = resource_manager.allocate_patient(
        patient_id="P002",
        category=TriageCategory.YELLOW,
    )

    assert next_allocation is not None
    assert next_allocation.ambulance_id == "A001"

