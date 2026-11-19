import re

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
)
from sceneops.agents.conversation_memory import PatientConversationManager
from sceneops.agents.disclosure import ConversationTurn
from sceneops.agents.disclosure_rules import (
    DisclosurePacketBuilder,
    PatientDisclosureRegistry,
)
from sceneops.agents.interfaces import PatientDialogueAgent, SafetyReviewer
from sceneops.agents.pain_consistency import (
    PainConsistencyValidator,
    PainState,
)
from sceneops.simulator.patients import Patient


class PatientConversationController:
    """Coordinates patient dialogue, safety review, and conversation memory."""

    _PAIN_VALUE_PATTERN = re.compile(r"^(\d+)/10$")

    def __init__(
        self,
        *,
        patient_agent: PatientDialogueAgent,
        safety_reviewer: SafetyReviewer,
        conversation_manager: PatientConversationManager,
        pain_validator: PainConsistencyValidator | None = None,
        rejection_message: str = (
            "I can't provide that response. Please continue the simulation "
            "using the available information."
        ),
    ) -> None:
        self.patient_agent = patient_agent
        self.safety_reviewer = safety_reviewer
        self.conversation_manager = conversation_manager
        self.pain_validator = pain_validator or PainConsistencyValidator()
        self.rejection_message = rejection_message

    @staticmethod
    def _get_authorized_pain(packet) -> PainState | None:
        for disclosure in packet.current_disclosures:
            if disclosure.fact != "reported_pain_level":
                continue

            match = PatientConversationController._PAIN_VALUE_PATTERN.fullmatch(
                disclosure.value
            )

            if match is None:
                return None

            return PainState(value=int(match.group(1)))

        return None

    async def handle(
        self,
        *,
        patient: Patient,
        user_message: str,
        disclosure_registry: PatientDisclosureRegistry,
        disclose_age: bool = False,
    ) -> AgentResponse:
        history = self.conversation_manager.get_history(patient.patient_id)

        packet = DisclosurePacketBuilder.build(
            patient,
            disclosure_registry=disclosure_registry,
            disclose_age=disclose_age,
            conversation_memory=history,
        )

        request = AgentRequest(
            task=AgentTask.PATIENT_DIALOGUE,
            user_message=user_message,
            patient_packet=packet,
        )

        response = await self.patient_agent.generate(request)

        if response.task != AgentTask.PATIENT_DIALOGUE:
            raise ValueError("Patient agent returned an unexpected response task.")

        review = await self.safety_reviewer.review(request, response)

        if not review.approved:
            return AgentResponse(
                task=AgentTask.PATIENT_DIALOGUE,
                agent_name="safety_reviewer",
                content=self.rejection_message,
            )

        pain_result = self.pain_validator.validate(
            response=response.content,
            current_pain=self._get_authorized_pain(packet),
        )

        if not pain_result.approved:
            return AgentResponse(
                task=AgentTask.PATIENT_DIALOGUE,
                agent_name="pain_consistency_validator",
                content=self.rejection_message,
            )

        self.conversation_manager.add_turn(
            patient.patient_id,
            ConversationTurn(
                user_message=user_message,
                patient_response=response.content,
            ),
        )

        return response
