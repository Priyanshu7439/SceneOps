from sceneops.agents.backend import TextGenerationBackend
from sceneops.agents.contracts import AgentRequest, AgentResponse, AgentTask


class PatientDialogueAgent:
    """Generate patient speech from a restricted DisclosurePacket only."""

    def __init__(self, backend: TextGenerationBackend) -> None:
        self.backend = backend

    async def generate(self, request: AgentRequest) -> AgentResponse:
        if request.task != AgentTask.PATIENT_DIALOGUE:
            raise ValueError(
                "PatientDialogueAgent only accepts patient_dialogue requests."
            )
        packet = request.patient_packet
        if packet is None:
            raise ValueError("Patient dialogue requires a disclosure packet.")

        profile = packet.profile
        system_prompt = "\n".join(
            [
                "You are role-playing one synthetic patient in an educational triage simulation.",
                "Use only the patient profile, current disclosures, and conversation memory below.",
                "Never infer or invent symptoms, vital signs, injuries, diagnoses, triage categories, or medical history.",
                "Do not provide medical advice or tell the learner what treatment to perform.",
                "If asked about something not disclosed, respond naturally that you do not know, cannot remember, or have not mentioned it.",
                "Stay in character and answer the learner's latest message concisely.",
                f"Patient age: {profile.age if profile.age is not None else 'not disclosed'}",
                f"Personality: {profile.personality or 'neutral'}",
                f"Speech style: {profile.speech_style or 'plain'}",
                "Current disclosures:",
                *[f"- {fact.fact}: {fact.value}" for fact in packet.current_disclosures],
                "Previously disclosed conversation facts:",
                *[f"- {fact.fact}: {fact.value}" for fact in packet.conversation_memory],
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
            task=AgentTask.PATIENT_DIALOGUE,
            agent_name="patient_dialogue_agent",
            content=content,
        )
