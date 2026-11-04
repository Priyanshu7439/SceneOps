
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from sceneops.simulator.resources import AllocationResult, ResourceManager


class TransportStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TransportRecord(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    transport_id: str
    ambulance_id: str
    patient_id: str
    started_at_seconds: int = Field(ge=0)
    travel_duration_seconds: int = Field(gt=0)
    completed_at_seconds: int | None = None
    status: TransportStatus = TransportStatus.IN_PROGRESS


class TransportManager:
    """Tracks ambulance journeys and releases resources on completion."""

    def __init__(self, resource_manager: ResourceManager) -> None:
        self.resource_manager = resource_manager
        self.transports: dict[str, TransportRecord] = {}
        self._next_transport_number = 1
        self._last_processed_second = 0

    def start_transport(
        self,
        allocation: AllocationResult,
        current_time_seconds: int,
        travel_duration_seconds: int,
    ) -> TransportRecord:
        if current_time_seconds < 0:
            raise ValueError("Current time cannot be negative.")

        if travel_duration_seconds <= 0:
            raise ValueError("Travel duration must be greater than zero.")

        if current_time_seconds < self._last_processed_second:
            raise ValueError("Transport time cannot move backwards.")

        ambulance = self.resource_manager.ambulances.get(
            allocation.ambulance_id
        )

        if ambulance is None:
            raise KeyError(
                f"Unknown ambulance: {allocation.ambulance_id}"
            )

        if ambulance.assigned_patient_id != allocation.patient_id:
            raise ValueError(
                "Ambulance is not assigned to this patient."
            )

        if any(
            record.patient_id == allocation.patient_id
            and record.status == TransportStatus.IN_PROGRESS
            for record in self.transports.values()
        ):
            raise ValueError("Patient already has an active transport.")

        if any(
            record.ambulance_id == allocation.ambulance_id
            and record.status == TransportStatus.IN_PROGRESS
            for record in self.transports.values()
        ):
            raise ValueError("Ambulance already has an active transport.")

        transport_id = f"TR-{self._next_transport_number:03d}"
        self._next_transport_number += 1

        record = TransportRecord(
            transport_id=transport_id,
            ambulance_id=allocation.ambulance_id,
            patient_id=allocation.patient_id,
            started_at_seconds=current_time_seconds,
            travel_duration_seconds=travel_duration_seconds,
        )

        self.transports[transport_id] = record
        return record

    def advance_time(
        self,
        current_time_seconds: int,
    ) -> list[TransportRecord]:
        if current_time_seconds < self._last_processed_second:
            raise ValueError("Transport time cannot move backwards.")

        completed: list[TransportRecord] = []

        for record in self.transports.values():
            if record.status != TransportStatus.IN_PROGRESS:
                continue

            completion_time = (
                record.started_at_seconds
                + record.travel_duration_seconds
            )

            if completion_time > current_time_seconds:
                continue

            self.resource_manager.release_ambulance(
                record.ambulance_id
            )

            record.completed_at_seconds = completion_time
            record.status = TransportStatus.COMPLETED
            completed.append(record)

        self._last_processed_second = current_time_seconds
        return completed
