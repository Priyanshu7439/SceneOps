
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from sceneops.simulator.triage import TriageCategory, TriageResult


class AmbulanceStatus(str, Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"


class Ambulance(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    ambulance_id: str = Field(min_length=1)
    assigned_patient_id: str | None = None

    @property
    def status(self) -> AmbulanceStatus:
        if self.assigned_patient_id is None:
            return AmbulanceStatus.AVAILABLE
        return AmbulanceStatus.OCCUPIED


class AllocationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ambulance_id: str
    patient_id: str
    triage_category: TriageCategory
    reason: str


class ResourceManager:
    """Manages ambulance availability and patient assignments."""

    PRIORITY = {
        TriageCategory.RED: 0,
        TriageCategory.YELLOW: 1,
        TriageCategory.GREEN: 2,
    }

    def __init__(self, ambulances: list[Ambulance]) -> None:
        ambulance_ids = [ambulance.ambulance_id for ambulance in ambulances]

        if len(ambulance_ids) != len(set(ambulance_ids)):
            raise ValueError("Ambulance IDs must be unique.")

        self.ambulances = {
            ambulance.ambulance_id: ambulance
            for ambulance in ambulances
        }

    def available_ambulances(self) -> list[Ambulance]:
        return [
            ambulance
            for ambulance in self.ambulances.values()
            if ambulance.status == AmbulanceStatus.AVAILABLE
        ]

    def allocate_patient(
        self,
        patient_id: str,
        category: TriageCategory,
    ) -> AllocationResult | None:
        """Assign the first available ambulance to one eligible patient."""

        if category not in self.PRIORITY:
            return None

        if any(
            ambulance.assigned_patient_id == patient_id
            for ambulance in self.ambulances.values()
        ):
            raise ValueError(f"Patient {patient_id} is already assigned.")

        available = self.available_ambulances()

        if not available:
            return None

        ambulance = available[0]
        ambulance.assigned_patient_id = patient_id

        return AllocationResult(
            ambulance_id=ambulance.ambulance_id,
            patient_id=patient_id,
            triage_category=category,
            reason=f"Assigned available ambulance to {category.value} patient.",
        )

    def allocate_next(
        self,
        waiting_patients: list[TriageResult],
    ) -> AllocationResult | None:
        """
        Allocate one ambulance to the highest-priority eligible patient.

        Python's stable sorting preserves arrival order for equal priorities.
        """

        eligible = [
            patient
            for patient in waiting_patients
            if patient.category in self.PRIORITY
        ]

        if not eligible:
            return None

        ordered = sorted(
            eligible,
            key=lambda patient: self.PRIORITY[patient.category],
        )

        next_patient = ordered[0]

        return self.allocate_patient(
            patient_id=next_patient.patient_id,
            category=next_patient.category,
        )

    def release_ambulance(self, ambulance_id: str) -> str:
        """Release an occupied ambulance and return its former patient ID."""

        ambulance = self.ambulances.get(ambulance_id)

        if ambulance is None:
            raise KeyError(f"Unknown ambulance: {ambulance_id}")

        if ambulance.assigned_patient_id is None:
            raise ValueError(f"Ambulance {ambulance_id} is already available.")

        patient_id = ambulance.assigned_patient_id
        ambulance.assigned_patient_id = None

        return patient_id
