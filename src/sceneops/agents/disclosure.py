from pydantic import BaseModel, ConfigDict, Field


class DisclosureFact(BaseModel):
    """A single fact explicitly permitted for disclosure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact: str = Field(min_length=1)
    value: str = Field(min_length=1)


class PatientProfile(BaseModel):
    """Non-clinical characteristics permitted for dialogue generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    age: int | None = Field(default=None, ge=0, le=120)
    personality: str | None = None
    speech_style: str | None = None


class DisclosurePacket(BaseModel):
    """
    Restricted information provided to a patient dialogue agent.

    Never construct this packet by dumping the internal Patient model.
    Every fact must pass an explicit disclosure rule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str = Field(min_length=1)
    profile: PatientProfile
    current_disclosures: list[DisclosureFact] = Field(default_factory=list)
    conversation_memory: list[DisclosureFact] = Field(default_factory=list)
    response_instructions: list[str] = Field(
        default_factory=lambda: [
            "Respond as the simulated patient.",
            "Only use information available in this packet.",
            "Do not invent medical facts.",
            "Do not reveal information that has not been disclosed.",
        ]
    )