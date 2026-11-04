
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class InjuryType(str, Enum):
    TRAUMA = "trauma"
    BURN = "burn"
    FRACTURE = "fracture"
    BLEEDING = "bleeding"
    RESPIRATORY = "respiratory"
    OTHER = "other"


class MentalStatus(str, Enum):
    ALERT = "alert"
    CONFUSED = "confused"
    UNRESPONSIVE = "unresponsive"


class Patient(BaseModel):
    """
    Internal patient state used by the deterministic simulator.

    IMPORTANT:
    This model contains internal simulation data.
    Never pass model_dump() directly to the patient LLM.
    Use the disclosure layer to expose permitted information.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    patient_id: str = Field(min_length=1)
    age: int = Field(ge=0, le=120)

    injury_type: InjuryType
    mental_status: MentalStatus = MentalStatus.ALERT

    # Simulation state
    ambulatory: bool
    respiratory_rate: int | None = Field(default=None, ge=0, le=80)
    radial_pulse_present: bool
    follows_commands: bool
    major_hemorrhage: bool

    # Patient characteristics for dialogue generation
    personality: str = Field(default="neutral", min_length=1)
    speech_style: str = Field(default="plain", min_length=1)

    # Subjective symptom value, not a clinical measurement
    pain_level: int = Field(default=0, ge=0, le=10)

    def update_pain(self, new_level: int) -> None:
        """Update the patient's simulated pain level."""
        self.pain_level = new_level
