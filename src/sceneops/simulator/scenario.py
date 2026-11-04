
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sceneops.simulator.deterioration import DeteriorationEvent
from sceneops.simulator.patients import Patient
from sceneops.simulator.resources import Ambulance


class ScenarioDefinition(BaseModel):
    """Defines a complete, reproducible simulation scenario."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)

    duration_seconds: int = Field(gt=0)

    patients: list[Patient] = Field(min_length=1)
    ambulances: list[Ambulance] = Field(min_length=1)

    deterioration_events: list[DeteriorationEvent] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_scenario(self) -> "ScenarioDefinition":
        patient_ids = [patient.patient_id for patient in self.patients]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Patient IDs must be unique within a scenario.")

        ambulance_ids = [
            ambulance.ambulance_id for ambulance in self.ambulances
        ]

        if len(ambulance_ids) != len(set(ambulance_ids)):
            raise ValueError("Ambulance IDs must be unique within a scenario.")

        patient_id_set = set(patient_ids)

        for event in self.deterioration_events:
            if event.patient_id not in patient_id_set:
                raise ValueError(
                    f"Deterioration event '{event.event_id}' references "
                    f"unknown patient '{event.patient_id}'."
                )

            if event.at_seconds > self.duration_seconds:
                raise ValueError(
                    f"Deterioration event '{event.event_id}' occurs "
                    "after the scenario duration."
                )

        return self
