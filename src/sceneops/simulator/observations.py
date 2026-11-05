
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ObservationType(str, Enum):
    RESPIRATORY_RATE = "respiratory_rate"
    RADIAL_PULSE = "radial_pulse"
    CAPILLARY_REFILL = "capillary_refill"
    MENTAL_STATUS = "mental_status"
    AMBULATORY_STATUS = "ambulatory_status"
    VISIBLE_HEMORRHAGE = "visible_hemorrhage"
    REPORTED_PAIN = "reported_pain"
    REPORTED_SYMPTOM = "reported_symptom"


class LearnerObservation(BaseModel):
    """A finding recorded through a learner assessment action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str = Field(min_length=1)
    observation_type: ObservationType
    value: str = Field(min_length=1)
    observed_at_seconds: int = Field(ge=0)


class LearnerObservationRegistry:
    """Stores learner observations separately from internal patient state."""

    def __init__(self) -> None:
        self._observations: dict[str, list[LearnerObservation]] = {}

    def record(self, observation: LearnerObservation) -> None:
        patient_observations = self._observations.setdefault(
            observation.patient_id,
            [],
        )
        patient_observations.append(observation)

    def get_for_patient(
        self,
        patient_id: str,
    ) -> list[LearnerObservation]:
        return list(self._observations.get(patient_id, []))

    def get_all(self) -> list[LearnerObservation]:
        return [
            observation
            for observations in self._observations.values()
            for observation in observations
        ]

    def clear_patient(self, patient_id: str) -> None:
        self._observations.pop(patient_id, None)
