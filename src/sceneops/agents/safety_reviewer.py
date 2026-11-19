import re

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    SafetyReview,
)


class RuleBasedSafetyReviewer:
    """Conservative baseline reviewer, not a clinical-grade classifier.

    Only current simulator-approved disclosures and scene facts are
    treated as evidence. Generated conversation history is never
    treated as verified fact.
    """

    _UNSAFE_PATTERNS = (
        re.compile(
            r"\b(?:heart attack|internal bleeding|spinal injury)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdiagnos(?:is|ed|ing)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:take|administer|give)\b.{0,50}\bmedication\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:perform surgery|intubat(?:e|ion)|triage category)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:red|yellow|green|black)[ -]tagged\b",
            re.IGNORECASE,
        ),
    )

    _VITAL_PATTERNS = (
        (
            "respiratory_rate",
            re.compile(
                r"\b(?:respiratory rate|breathing rate)\s*"
                r"(?:is|of|:)?\s*(\d+)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "respiratory_rate",
            re.compile(
                r"\b(\d+)\s*(?:breaths per minute|bpm)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "blood_pressure",
            re.compile(
                r"\b(?:blood pressure|bp)\s*"
                r"(?:is|of|:)?\s*(\d{2,3}\s*/\s*\d{2,3})",
                re.IGNORECASE,
            ),
        ),
        (
            "oxygen_saturation",
            re.compile(
                r"\b(?:oxygen saturation|spo2)\s*"
                r"(?:is|of|:)?\s*(\d{1,3})\s*%",
                re.IGNORECASE,
            ),
        ),
        (
            "capillary_refill",
            re.compile(
                r"\bcapillary refill\s*(?:is|of|:)?\s*"
                r"(\d+(?:\.\d+)?)\s*(?:seconds|secs|s)\b",
                re.IGNORECASE,
            ),
        ),
    )

    _PATIENT_ID_PATTERN = re.compile(
        r"\b(P[A-Za-z0-9_-]{0,63})\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _normalize_value(value: str) -> str:
        return re.sub(r"\s+", "", value).casefold()

    @classmethod
    def _scene_evidence(cls, request: AgentRequest) -> dict:
        evidence = {}

        for fact in request.scene_facts:
            match = re.fullmatch(
                r"patient_observation:"
                r"([A-Za-z0-9_-]{1,64}):"
                r"([a-z_]+)",
                fact.fact,
            )

            if match is None:
                continue

            patient_id, observation_type = match.groups()

            value_match = re.fullmatch(
                r"(.+?)\s+\(observed at simulation second \d+\)",
                fact.value,
            )

            if value_match is None:
                continue

            key = (patient_id.casefold(), observation_type)
            evidence[key] = cls._normalize_value(value_match.group(1))

        return evidence

    @classmethod
    def _review_scene_measurements(
        cls,
        request: AgentRequest,
        response: AgentResponse,
    ) -> list[str]:
        evidence = cls._scene_evidence(request)
        reasons = []

        for observation_type, pattern in cls._VITAL_PATTERNS:
            for match in pattern.finditer(response.content):
                claimed_value = cls._normalize_value(match.group(1))

                preceding_text = response.content[
                    max(0, match.start() - 80) : match.start()
                ]

                patient_matches = list(cls._PATIENT_ID_PATTERN.finditer(preceding_text))

                if not patient_matches:
                    reasons.append(
                        "Scene narration contains a clinical measurement "
                        "without an identifiable patient."
                    )
                    continue

                patient_id = patient_matches[-1].group(1).casefold()
                approved_value = evidence.get((patient_id, observation_type))

                if approved_value is None or claimed_value != approved_value:
                    reasons.append(
                        "Scene narration contains a clinical measurement "
                        "that does not match the approved observation "
                        "for that patient."
                    )

        return reasons

    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview:
        reasons: list[str] = []

        if response.task != request.task:
            return SafetyReview(
                approved=False,
                reasons=("Response task does not match request task.",),
            )

        text = response.content.casefold()

        for pattern in self._UNSAFE_PATTERNS:
            if pattern.search(text):
                reasons.append(
                    "Response contains diagnosis, treatment, or triage "
                    "decision language."
                )
                break

        if request.task == AgentTask.PATIENT_DIALOGUE:
            packet = request.patient_packet

            if packet is not None:
                disclosed_text = " ".join(
                    fact.value for fact in packet.current_disclosures
                ).casefold()

                for _, pattern in self._VITAL_PATTERNS:
                    for match in pattern.finditer(response.content):
                        claimed_value = self._normalize_value(match.group(1))

                        if claimed_value not in self._normalize_value(disclosed_text):
                            reasons.append(
                                "Response appears to introduce an "
                                "undisclosed clinical measurement."
                            )
                            break

                    if any(
                        reason.startswith("Response appears to introduce")
                        for reason in reasons
                    ):
                        break

        elif request.task == AgentTask.SCENE_NARRATION:
            reasons.extend(self._review_scene_measurements(request, response))

        return SafetyReview(
            approved=not reasons,
            reasons=tuple(reasons),
        )
