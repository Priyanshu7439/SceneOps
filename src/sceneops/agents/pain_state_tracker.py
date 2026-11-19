from pydantic import BaseModel, ConfigDict

from sceneops.agents.pain_consistency import PainState
from sceneops.simulator.patients import Patient


class PainStateTransition(BaseModel):
    """Previous and current simulator-authorized pain states."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    previous: PainState
    current: PainState

    @property
    def changed(self) -> bool:
        return self.previous.value != self.current.value

    @property
    def direction(self) -> str:
        if self.current.value > self.previous.value:
            return "increase"

        if self.current.value < self.previous.value:
            return "decrease"

        return "unchanged"


class PainStateTracker:
    """Tracks simulator pain state independently for each patient.

    The first observation establishes the baseline. Later observations
    compare the current simulator value with the previous observation.
    """

    def __init__(self) -> None:
        self._last_observed: dict[str, PainState] = {}

    def observe(self, patient: Patient) -> PainStateTransition:
        if not isinstance(patient, Patient):
            raise TypeError("patient must be a Patient instance.")

        current = PainState(value=patient.pain_level)

        previous = self._last_observed.get(
            patient.patient_id,
            current,
        )

        transition = PainStateTransition(
            previous=previous,
            current=current,
        )

        self._last_observed[patient.patient_id] = current

        return transition

    def clear_patient(self, patient_id: str) -> None:
        self._last_observed.pop(patient_id, None)

    def clear_all(self) -> None:
        self._last_observed.clear()
