import json
import re
from enum import Enum

from sceneops.agents.backend import TextGenerationBackend
from sceneops.agents.contracts import AgentRequest, AgentResponse, AgentTask


class DialogueStyle(str, Enum):
    PLAIN = "plain"
    HESITANT = "hesitant"
    BRIEF = "brief"
    DESCRIPTIVE = "descriptive"
    REPETITIVE = "repetitive"


class PatientDialogueAgent:
    """Generate patient dialogue using simulator-approved facts.

    The LLM selects a presentation style only.
    Python determines the answer and renders all factual content.
    """

    _PAIN_QUESTION = re.compile(
        r"\b(pain|hurt|hurting|ache|aching)\b",
        re.IGNORECASE,
    )

    _AGE_QUESTION = re.compile(
        r"\b(how old|your age|age are you)\b",
        re.IGNORECASE,
    )

    _STYLE_PROMPT = """
You select a conversational presentation style for a synthetic patient.

You must return exactly one JSON object:
{"style": "plain"}
{"style": "hesitant"}
{"style": "brief"}
{"style": "descriptive"}
{"style": "repetitive"}

Choose only one of these styles:
- plain: normal, straightforward wording
- hesitant: hesitant expressions and pauses
- brief: short and concise wording
- descriptive: more elaborate conversational wording
- repetitive: natural repetition of words or phrases

Use the patient's personality, speech style, and learner message
to select the most appropriate presentation style.

Do not generate dialogue.
Do not generate symptoms, medical facts, measurements, or diagnoses.
Do not include explanations or additional JSON fields.
"""

    def __init__(self, backend: TextGenerationBackend) -> None:
        self.backend = backend

    @staticmethod
    def _get_style(raw_response: str) -> DialogueStyle:
        if not isinstance(raw_response, str):
            return DialogueStyle.PLAIN

        raw_response = raw_response.strip()

        if raw_response.startswith("```") and raw_response.endswith("```"):
            lines = raw_response.splitlines()
            raw_response = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(raw_response)

            if not isinstance(data, dict):
                return DialogueStyle.PLAIN

            if set(data) != {"style"}:
                return DialogueStyle.PLAIN

            return DialogueStyle(data["style"])

        except (ValueError, TypeError):
            return DialogueStyle.PLAIN

    @staticmethod
    def _get_disclosure(packet, fact_name: str) -> str | None:
        for fact in packet.current_disclosures:
            if fact.fact == fact_name:
                return fact.value

        return None

    @staticmethod
    def _render_pain(value: str, style: DialogueStyle) -> str:
        if style == DialogueStyle.HESITANT:
            return f"Um... I think my pain is around {value}."

        if style == DialogueStyle.BRIEF:
            return f"My pain is {value}."

        if style == DialogueStyle.DESCRIPTIVE:
            return f"I'd describe my pain as approximately {value}."

        if style == DialogueStyle.REPETITIVE:
            return f"My pain is {value}... that's how much it hurts."

        return f"My pain is around {value}."

    @staticmethod
    def _render_age(value: int, style: DialogueStyle) -> str:
        if style == DialogueStyle.HESITANT:
            return f"Um... I'm {value} years old."

        if style == DialogueStyle.BRIEF:
            return f"I'm {value}."

        if style == DialogueStyle.DESCRIPTIVE:
            return f"I'm {value} years old, as far as I know."

        if style == DialogueStyle.REPETITIVE:
            return f"I'm {value} years old... {value} years old."

        return f"I'm {value} years old."

    @staticmethod
    def _render_unknown(style: DialogueStyle) -> str:
        if style == DialogueStyle.HESITANT:
            return "I'm not sure... nobody has told me that."

        if style == DialogueStyle.BRIEF:
            return "I don't know."

        if style == DialogueStyle.DESCRIPTIVE:
            return "I don't have that information, so I can't tell you."

        if style == DialogueStyle.REPETITIVE:
            return "I don't know... I really don't know."

        return "I don't know. Nobody has told me that."

    async def generate(self, request: AgentRequest) -> AgentResponse:
        if request.task != AgentTask.PATIENT_DIALOGUE:
            raise ValueError(
                "PatientDialogueAgent only accepts patient_dialogue requests."
            )

        packet = request.patient_packet

        if packet is None:
            raise ValueError("Patient dialogue requires a disclosure packet.")

        raw_style = await self.backend.generate(
            system_prompt=self._STYLE_PROMPT,
            user_prompt=(
                f"Personality: {packet.profile.personality or 'neutral'}\n"
                f"Speech style: {packet.profile.speech_style or 'plain'}\n"
                f"Learner message: {request.user_message}"
            ),
        )

        style = self._get_style(raw_style)

        if self._PAIN_QUESTION.search(request.user_message):
            pain_value = self._get_disclosure(
                packet,
                "reported_pain_level",
            )

            if pain_value is not None:
                content = self._render_pain(pain_value, style)
            else:
                content = self._render_unknown(style)

        elif self._AGE_QUESTION.search(request.user_message):
            if packet.profile.age is not None:
                content = self._render_age(packet.profile.age, style)
            else:
                content = self._render_unknown(style)

        else:
            content = self._render_unknown(style)

        return AgentResponse(
            task=AgentTask.PATIENT_DIALOGUE,
            agent_name="patient_dialogue_agent",
            content=content,
        )
