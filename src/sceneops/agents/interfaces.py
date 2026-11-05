from typing import Protocol

from sceneops.agents.contracts import AgentRequest, AgentResponse, SafetyReview


class PatientDialogueAgent(Protocol):
    """Generates patient speech using only the restricted disclosure packet."""

    async def generate(self, request: AgentRequest) -> AgentResponse: ...


class SceneNarrationAgent(Protocol):
    """Narrates a scene using only simulator-approved scene facts."""

    async def generate(self, request: AgentRequest) -> AgentResponse: ...


class SafetyReviewer(Protocol):
    """Reviews generated text before it is returned to the application."""

    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview: ...
