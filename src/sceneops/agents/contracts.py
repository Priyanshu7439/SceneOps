from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sceneops.agents.disclosure import DisclosurePacket


class AgentTask(str, Enum):
    PATIENT_DIALOGUE = "patient_dialogue"
    SCENE_NARRATION = "scene_narration"


class SceneFact(BaseModel):
    """A simulator-approved, learner-visible fact for scene narration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact: str = Field(min_length=1)
    value: str = Field(min_length=1)


class AgentRequest(BaseModel):
    """Typed input contract shared by the agent orchestration layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: AgentTask
    user_message: str = Field(min_length=1, max_length=4000)
    patient_packet: DisclosurePacket | None = None
    scene_facts: tuple[SceneFact, ...] = ()

    @model_validator(mode="after")
    def validate_task_payload(self) -> "AgentRequest":
        if self.task == AgentTask.PATIENT_DIALOGUE:
            if self.patient_packet is None:
                raise ValueError("Patient dialogue requires a disclosure packet.")
            if self.scene_facts:
                raise ValueError("Patient dialogue cannot receive scene facts.")
        elif self.task == AgentTask.SCENE_NARRATION:
            if self.patient_packet is not None:
                raise ValueError("Scene narration cannot receive a patient packet.")
            if not self.scene_facts:
                raise ValueError("Scene narration requires approved scene facts.")
        return self


class AgentResponse(BaseModel):
    """Model-agnostic response returned by a specialized agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: AgentTask
    agent_name: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=8000)


class SafetyReview(BaseModel):
    """Result from the safety reviewer; rejected content is never returned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    reasons: tuple[str, ...] = ()
