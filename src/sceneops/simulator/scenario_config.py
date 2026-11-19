from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from sceneops.simulator.deterioration import PatientChanges


class PatientArchetype(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    BLACK = "black"
    PENDING = "pending"


class ScenarioEventSpec(BaseModel):
    """Configuration for one seeded deterioration event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    target_archetype: PatientArchetype
    earliest_seconds: int = Field(ge=1)
    latest_seconds: int = Field(ge=1)
    changes: PatientChanges

    @model_validator(mode="after")
    def validate_time_window(self):
        if self.latest_seconds < self.earliest_seconds:
            raise ValueError("latest_seconds must be >= earliest_seconds.")
        return self


class ScenarioConfig(BaseModel):
    """Reproducible configuration for hybrid scenario generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    seed: int = Field(ge=0)

    patient_count: int = Field(default=20, ge=1)
    ambulance_count: int = Field(default=5, ge=1)
    duration_seconds: int = Field(default=600, gt=0)

    archetype_counts: dict[PatientArchetype, int] = Field(
        default_factory=lambda: {
            PatientArchetype.GREEN: 4,
            PatientArchetype.YELLOW: 4,
            PatientArchetype.RED: 4,
            PatientArchetype.BLACK: 4,
            PatientArchetype.PENDING: 4,
        }
    )

    event_specs: tuple[ScenarioEventSpec, ...] = ()

    @model_validator(mode="after")
    def validate_configuration(self):
        expected_archetypes = set(PatientArchetype)
        configured_archetypes = set(self.archetype_counts)

        if configured_archetypes != expected_archetypes:
            raise ValueError("All patient archetypes must be explicitly configured.")
        if sum(self.archetype_counts.values()) != self.patient_count:
            raise ValueError("Archetype counts must sum to patient_count.")

        if any(count < 0 for count in self.archetype_counts.values()):
            raise ValueError("Archetype counts cannot be negative.")

        event_ids = [event.event_id for event in self.event_specs]

        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Scenario event IDs must be unique.")

        for event in self.event_specs:
            if event.latest_seconds > self.duration_seconds:
                raise ValueError(f"Event {event.event_id} exceeds scenario duration.")

            if self.archetype_counts.get(event.target_archetype, 0) == 0:
                raise ValueError(
                    f"Event {event.event_id} targets an archetype "
                    "with no configured patients."
                )

        return self
