from sceneops.agents.contracts import AgentRequest, AgentResponse, AgentTask
from sceneops.agents.patient_dialogue import TextGenerationBackend


class SceneNarrationAgent:
    """Narrate only simulator-approved, learner-visible scene facts."""

    def __init__(self, backend: TextGenerationBackend) -> None:
        self.backend = backend

    async def generate(self, request: AgentRequest) -> AgentResponse:
        if request.task != AgentTask.SCENE_NARRATION:
            raise ValueError("SceneNarrationAgent only accepts scene_narration requests.")

        facts = "\n".join(
            f"- {fact.fact}: {fact.value}" for fact in request.scene_facts
        )

        system_prompt = "\n".join(  # noqa: FLY002
            [
                "You narrate a synthetic mass-casualty training scene.",
                "Use only the approved scene facts supplied below.",
                "Do not invent patients, injuries, vital signs, resources, events, diagnoses, or triage decisions.",
                "Do not provide clinical instructions or make medical decisions.",
                "Describe the scene clearly and neutrally for the learner.",
                "Approved scene facts:",
                facts,
            ]
        )
        content = await self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=request.user_message,
        )
        content = content.strip()
        if not content:
            raise ValueError("Text generation backend returned an empty response.")

        return AgentResponse(
            task=AgentTask.SCENE_NARRATION,
            agent_name="scene_narration_agent",
            content=content,
        )
