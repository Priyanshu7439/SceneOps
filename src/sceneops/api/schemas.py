from pydantic import BaseModel, ConfigDict, Field

from sceneops.agents.disclosure_rules import PatientQuestion
from sceneops.simulator.observations import ObservationType


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_index: int = Field(default=1, ge=1)


class CreateSessionResponse(BaseModel):
    session_id: str
    scenario_id: str
    title: str
    description: str
    patient_count: int
    status: str


class SessionSnapshotResponse(BaseModel):
    session_id: str
    status: str
    clock: dict[str, int]
    patient_count: int
    assessed_patient_count: int
    allocated_patient_count: int
    available_ambulance_count: int
    event_count: int


class PatientDialogueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    disclose_age: bool = False


class PatientDialogueResponse(BaseModel):
    patient_id: str
    agent_name: str
    content: str


class PatientQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: PatientQuestion


class PatientQuestionResponse(BaseModel):
    patient_id: str
    fact: str
    value: str


class AdvanceTimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seconds: int = Field(default=60, ge=1)


class ClockResponse(BaseModel):
    elapsed_seconds: int
    tick_count: int


class MessageResponse(BaseModel):
    message: str
    status: str

class PatientAssessmentResponse(BaseModel):
    patient_id: str
    category: str
    reason: str
    criteria: list[str]


class AllocationResponse(BaseModel):
    patient_id: str
    ambulance_id: str
    triage_category: str
    reason: str


class StartTransportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    travel_duration_seconds: int = Field(default=300, ge=1)


class TransportResponse(BaseModel):
    transport_id: str
    patient_id: str
    ambulance_id: str
    started_at_seconds: int
    travel_duration_seconds: int
    completed_at_seconds: int | None = None
    status: str

class PatientObservationRequest(BaseModel):
    """Request to observe a patient during a running simulation."""

    model_config = ConfigDict(extra="forbid")

    observation_type: ObservationType


class PatientObservationResponse(BaseModel):
    """Simulator-derived learner observation."""

    patient_id: str
    observation_type: ObservationType
    value: str
    observed_at_seconds: int


class SceneNarrationRequest(BaseModel):
    """Learner request for simulator-grounded scene narration."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)


class SceneNarrationResponse(BaseModel):
    """Safety-reviewed scene narration."""

    agent_name: str
    content: str


class TemporalSceneNarrationRequest(BaseModel):
    """Learner request for timestamped observation-history narration."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)


class TemporalSceneNarrationResponse(BaseModel):
    """Safety-reviewed temporal scene narration."""

    agent_name: str
    content: str
class PatientDashboardItem(BaseModel):
    patient_id: str
    assessment_status: str
    triage_category: str
    ambulance_id: str | None = None
    transport_status: str


class PatientListResponse(BaseModel):
    patients: list[PatientDashboardItem]

class AmbulanceDashboardItem(BaseModel):
    ambulance_id: str
    status: str
    assigned_patient_id: str | None = None


class AmbulanceListResponse(BaseModel):
    ambulances: list[AmbulanceDashboardItem]


class SimulationEventItem(BaseModel):
    event_type: str
    simulation_second: int
    description: str
    patient_id: str | None = None


class SimulationEventListResponse(BaseModel):
    events: list[SimulationEventItem]
