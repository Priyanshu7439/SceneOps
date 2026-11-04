
from enum import Enum

from pydantic import BaseModel, ConfigDict

from sceneops.simulator.clock import ClockSnapshot, SimulationClock
from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    DeteriorationManager,
)
from sceneops.simulator.patients import Patient
from sceneops.simulator.resources import (
    AllocationResult,
    Ambulance,
    ResourceManager,
)
from sceneops.simulator.triage import (
    TriageResult,
    classify_patient,
)


class SimulationStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class SimulationEvent(BaseModel):
    """Record of an important simulation action."""

    model_config = ConfigDict(frozen=True)

    event_type: str
    simulation_second: int
    description: str
    patient_id: str | None = None


class SimulationEngine:
    """
    Coordinates patients, triage, resources and simulation time.

    Medical classification remains inside the triage module.
    """

    def __init__(
        self,
        ambulances: list[Ambulance],
        clock: SimulationClock | None = None,
        deterioration_events: list[DeteriorationEvent] | None = None,
    ) -> None:
        self.status = SimulationStatus.CREATED
        self.clock = clock or SimulationClock()
        self.resource_manager = ResourceManager(ambulances)

        self.deterioration_manager = DeteriorationManager(
            deterioration_events or []
        )

        self.patients: dict[str, Patient] = {}
        self.triage_results: dict[str, TriageResult] = {}
        self.allocations: dict[str, AllocationResult] = {}
        self.events: list[SimulationEvent] = []

    def _record_event(
        self,
        event_type: str,
        description: str,
        patient_id: str | None = None,
    ) -> None:
        self.events.append(
            SimulationEvent(
                event_type=event_type,
                simulation_second=self.clock.elapsed_seconds,
                description=description,
                patient_id=patient_id,
            )
        )

    def add_patient(self, patient: Patient) -> None:
        """Register a patient before the simulation starts."""

        if self.status != SimulationStatus.CREATED:
            raise RuntimeError(
                "Patients can only be registered before simulation starts."
            )

        if patient.patient_id in self.patients:
            raise ValueError(
                f"Patient {patient.patient_id} is already registered."
            )

        self.patients[patient.patient_id] = patient

        self._record_event(
            event_type="patient_registered",
            description=f"Patient {patient.patient_id} registered.",
            patient_id=patient.patient_id,
        )

    def start(self) -> None:
        """Validate scenario references and start simulation."""

        if self.status != SimulationStatus.CREATED:
            raise RuntimeError("Only a created simulation can be started.")

        for event in self.deterioration_manager.events:
            if event.patient_id not in self.patients:
                raise ValueError(
                    f"Scheduled event {event.event_id} references "
                    f"unregistered patient {event.patient_id}."
                )

        self.status = SimulationStatus.RUNNING
        self._record_event(
            "simulation_started",
            "Simulation started.",
        )

    def pause(self) -> None:
        if self.status != SimulationStatus.RUNNING:
            raise RuntimeError("Only a running simulation can be paused.")

        self.status = SimulationStatus.PAUSED
        self._record_event(
            "simulation_paused",
            "Simulation paused.",
        )

    def resume(self) -> None:
        if self.status != SimulationStatus.PAUSED:
            raise RuntimeError("Only a paused simulation can be resumed.")

        self.status = SimulationStatus.RUNNING
        self._record_event(
            "simulation_resumed",
            "Simulation resumed.",
        )

    def complete(self) -> None:
        if self.status not in (
            SimulationStatus.RUNNING,
            SimulationStatus.PAUSED,
        ):
            raise RuntimeError(
                "Only a running or paused simulation can be completed."
            )

        self.status = SimulationStatus.COMPLETED
        self._record_event(
            "simulation_completed",
            "Simulation completed.",
        )

    def assess_patient(self, patient_id: str) -> TriageResult:
        """Run deterministic triage for a registered patient."""

        if self.status not in (
            SimulationStatus.CREATED,
            SimulationStatus.RUNNING,
        ):
            raise RuntimeError(
                "Patient assessment is unavailable in the current state."
            )

        patient = self.patients.get(patient_id)

        if patient is None:
            raise KeyError(f"Unknown patient: {patient_id}")

        result = classify_patient(patient)
        self.triage_results[patient_id] = result

        self._record_event(
            event_type="patient_assessed",
            description=f"Patient classified as {result.category.value}.",
            patient_id=patient_id,
        )

        return result

    def allocate_next_patient(self) -> AllocationResult | None:
        """Allocate an ambulance to the highest-priority waiting patient."""

        if self.status != SimulationStatus.RUNNING:
            raise RuntimeError(
                "Allocation requires a running simulation."
            )

        waiting = [
            result
            for patient_id, result in self.triage_results.items()
            if patient_id not in self.allocations
        ]

        result = self.resource_manager.allocate_next(waiting)

        if result is None:
            return None

        self.allocations[result.patient_id] = result

        self._record_event(
            event_type="ambulance_allocated",
            description=(
                f"Ambulance {result.ambulance_id} assigned to "
                f"{result.patient_id}."
            ),
            patient_id=result.patient_id,
        )

        return result

    def release_ambulance(self, ambulance_id: str) -> str:
        """Release an ambulance and remove its patient allocation."""

        if self.status != SimulationStatus.RUNNING:
            raise RuntimeError(
                "Ambulance release requires a running simulation."
            )

        patient_id = self.resource_manager.release_ambulance(
            ambulance_id
        )

        self.allocations.pop(patient_id, None)

        self._record_event(
            event_type="ambulance_released",
            description=f"Ambulance {ambulance_id} released.",
            patient_id=patient_id,
        )

        return patient_id

    def advance_time(self, seconds: int = 60) -> ClockSnapshot:
        """
        Advance simulation time, apply scheduled changes,
        and reassess affected patients.
        """

        if self.status != SimulationStatus.RUNNING:
            raise RuntimeError(
                "Clock can advance only while simulation is running."
            )

        snapshot = self.clock.advance(seconds)

        applied_events = self.deterioration_manager.apply_due_events(
            patients=self.patients,
            elapsed_seconds=snapshot.elapsed_seconds,
        )

        for applied in applied_events:
            self._record_event(
                event_type="patient_state_changed",
                description=(
                    f"Scenario event {applied.event_id} changed: "
                    f"{', '.join(applied.changed_fields)}."
                ),
                patient_id=applied.patient_id,
            )

            result = self.assess_patient(applied.patient_id)

            self._record_event(
                event_type="patient_reassessed",
                description=(
                    f"Updated triage category: "
                    f"{result.category.value}."
                ),
                patient_id=applied.patient_id,
            )

        self._record_event(
            event_type="clock_advanced",
            description=f"Simulation advanced by {seconds} seconds.",
        )

        return snapshot

    def get_snapshot(self) -> dict:
        """Return a summary of the current simulation state."""

        return {
            "status": self.status.value,
            "clock": {
                "elapsed_seconds": self.clock.elapsed_seconds,
                "tick_count": self.clock.tick_count,
            },
            "patient_count": len(self.patients),
            "assessed_patient_count": len(self.triage_results),
            "allocated_patient_count": len(self.allocations),
            "available_ambulance_count": len(
                self.resource_manager.available_ambulances()
            ),
            "event_count": len(self.events),
        }
