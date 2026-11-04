
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sceneops.simulator.patients import MentalStatus, Patient


class PatientChanges(BaseModel):
    """Explicit state changes defined by a simulation scenario."""

    model_config = ConfigDict(extra="forbid")

    respiratory_rate: int | None = Field(default=None, ge=0, le=80)
    ambulatory: bool | None = None
    radial_pulse_present: bool | None = None
    follows_commands: bool | None = None
    major_hemorrhage: bool | None = None
    mental_status: MentalStatus | None = None

    @model_validator(mode="after")
    def require_at_least_one_change(self):
        if not self.model_fields_set:
            raise ValueError("At least one patient change must be specified.")
        return self


class DeteriorationEvent(BaseModel):
    """A scenario-defined patient state change at a simulation timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    at_seconds: int = Field(ge=0)
    changes: PatientChanges


class AppliedDeterioration(BaseModel):
    """Record of a successfully applied scenario event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    patient_id: str
    simulation_second: int
    changed_fields: tuple[str, ...]


class DeteriorationManager:
    """
    Applies deterministic, scenario-authored patient state changes.

    This manager does not predict clinical deterioration.
    It only applies explicit changes defined in scenario data.
    """

    def __init__(self, events: list[DeteriorationEvent]) -> None:
        event_ids = [event.event_id for event in events]

        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Deterioration event IDs must be unique.")

        # Python's stable sort preserves original order for equal timestamps.
        self.events = sorted(events, key=lambda event: event.at_seconds)
        self._applied_event_ids: set[str] = set()
        self._last_processed_second = 0

    @property
    def applied_event_ids(self) -> frozenset[str]:
        return frozenset(self._applied_event_ids)

    def apply_due_events(
        self,
        patients: dict[str, Patient],
        elapsed_seconds: int,
    ) -> list[AppliedDeterioration]:
        """Apply all scheduled events due at or before the supplied time."""

        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, int):
            raise TypeError("elapsed_seconds must be an integer.")

        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative.")

        if elapsed_seconds < self._last_processed_second:
            raise ValueError("Simulation time cannot move backwards.")

        due_events = [
            event
            for event in self.events
            if (
                event.at_seconds <= elapsed_seconds
                and event.event_id not in self._applied_event_ids
            )
        ]

        # Validate all patient references before changing any patient.
        for event in due_events:
            if event.patient_id not in patients:
                raise KeyError(
                    f"Unknown patient {event.patient_id} "
                    f"for event {event.event_id}."
                )

        applied: list[AppliedDeterioration] = []

        for event in due_events:
            patient = patients[event.patient_id]
            changes = event.changes.model_dump(exclude_unset=True)

            for field_name, value in changes.items():
                setattr(patient, field_name, value)

            self._applied_event_ids.add(event.event_id)

            applied.append(
                AppliedDeterioration(
                    event_id=event.event_id,
                    patient_id=event.patient_id,
                    simulation_second=elapsed_seconds,
                    changed_fields=tuple(changes.keys()),
                )
            )

        self._last_processed_second = elapsed_seconds

        return applied
