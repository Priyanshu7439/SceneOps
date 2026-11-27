
import logging

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
)
from sceneops.agents.interfaces import (
    PatientDialogueAgent,
    SafetyReviewer,
    SceneNarrationAgent,
    TemporalSceneNarrationAgent,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Routes typed requests and returns only safety-approved agent output.

    This class deliberately contains no medical decision logic and no LLM
    provider code. Concrete model-backed agents and a real reviewer are
    injected later, keeping orchestration testable and provider-independent.
    """

    def __init__(
        self,
        *,
        patient_agent: PatientDialogueAgent,
        scene_agent: SceneNarrationAgent,
        safety_reviewer: SafetyReviewer,
        temporal_scene_agent: TemporalSceneNarrationAgent | None = None,
        rejection_message: str = (
            "I can't provide that response. Please continue the simulation "
            "using the available information."
        ),
    ) -> None:
        self.patient_agent = patient_agent
        self.scene_agent = scene_agent
        self.temporal_scene_agent = temporal_scene_agent
        self.safety_reviewer = safety_reviewer
        self.rejection_message = rejection_message

    async def handle(self, request: AgentRequest) -> AgentResponse:
        if request.task == AgentTask.PATIENT_DIALOGUE:
            response = await self.patient_agent.generate(request)

        elif request.task == AgentTask.SCENE_NARRATION:
            response = await self.scene_agent.generate(request)

        elif request.task == AgentTask.TEMPORAL_SCENE_NARRATION:
            if self.temporal_scene_agent is None:
                raise ValueError(
                    "Temporal scene narration is not configured."
                )
            response = await self.temporal_scene_agent.generate(request)

        else:
            raise ValueError(f"Unsupported agent task: {request.task}")

        if response.task != request.task:
            raise ValueError("Agent response task does not match request task.")

        review = await self.safety_reviewer.review(request, response)

        if not review.approved:
            logger.warning(
                "Safety review rejected task %s. Reasons: %s",
                request.task.value,
                "; ".join(review.reasons) or "No reasons provided.",
            )
            return AgentResponse(
                task=request.task,
                agent_name="safety_reviewer",
                content=self.rejection_message,
            )

        return response
