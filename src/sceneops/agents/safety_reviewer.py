import re

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    SafetyReview,
)


class RuleBasedSafetyReviewer:
    """Conservative baseline reviewer; not a clinical-grade safety classifier.

    It blocks explicit diagnosis/treatment language and common unsupported
    vital-sign claims. A later model-backed evaluator can supplement this
    deterministic baseline, but must not replace simulator-side controls.
    """

    _UNSAFE_PHRASES = (
        "you have a heart attack",
        "you are having a heart attack",
        "you have internal bleeding",
        "you have a spinal injury",
        "your diagnosis is",
        "you should take medication",
        "take this medication",
        "administer medication",
        "give them medication",
        "perform surgery",
        "intubate the patient",
        "your triage category is",
        "you are red-tagged",
        "you are yellow-tagged",
        "you are green-tagged",
        "you are black-tagged",
    )
    _VITAL_PATTERNS = (
        re.compile(r"\b(?:respiratory rate|breathing rate)\s*(?:is|of|:)?\s*\d+", re.IGNORECASE),
        re.compile(r"\b\d+\s*(?:breaths per minute|bpm)\b", re.IGNORECASE),
        re.compile(r"\b(?:blood pressure|bp)\s*(?:is|of|:)?\s*\d{2,3}\s*/\s*\d{2,3}", re.IGNORECASE),
        re.compile(r"\b(?:oxygen saturation|spo2)\s*(?:is|of|:)?\s*\d{1,3}\s*%", re.IGNORECASE),
        re.compile(r"\b(?:capillary refill)\s*(?:is|of|:)?\s*\d+(?:\.\d+)?\s*(?:seconds|secs|s)\b", re.IGNORECASE),
    )

    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview:
        reasons: list[str] = []
        text = response.content.casefold()

        for phrase in self._UNSAFE_PHRASES:
            if phrase in text:
                reasons.append("Response contains diagnosis, treatment, or triage instruction language.")
                break

        if request.task == AgentTask.PATIENT_DIALOGUE and request.patient_packet is not None:
            disclosed_text = " ".join(
                [
                    *(fact.value for fact in request.patient_packet.current_disclosures),
                    *(fact.value for fact in request.patient_packet.conversation_memory),
                ]
            ).casefold()
            for pattern in self._VITAL_PATTERNS:
                for match in pattern.finditer(response.content):
                    if match.group(0).casefold() not in disclosed_text:
                        reasons.append("Response appears to introduce an undisclosed clinical measurement.")
                        break
                if reasons and reasons[-1].startswith("Response appears"):
                    break

        return SafetyReview(approved=not reasons, reasons=tuple(reasons))
