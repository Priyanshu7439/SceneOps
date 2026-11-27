

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



    _TEMPORAL_PATTERNS = (

        (

            "respiratory_rate",

            re.compile(

                r"\b(?:respiratory rate|breathing rate)\s*"

                r"(?:is|was|of|:)?\s*(\d+)\b",

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

            "radial_pulse",

            re.compile(

                r"\b(?:radial pulse|pulse)\s*(?:is|was|of|:)?\s*"

                r"(\d+)\s*(?:bpm|beats per minute)\b",

                re.IGNORECASE,

            ),

        ),

        (

            "capillary_refill",

            re.compile(

                r"\bcapillary refill\s*(?:is|was|of|:)?\s*"

                r"(\d+(?:\.\d+)?)\s*(?:seconds|secs|s)\b",

                re.IGNORECASE,

            ),

        ),

    )



    _TEMPORAL_CATEGORICAL_PATTERNS = (

        (

            "radial_pulse",

            "present",

            re.compile(

                r"\bradial pulse\s+(?:is|was)\s+present\b",

                re.IGNORECASE,

            ),

        ),

        (

            "radial_pulse",

            "absent",

            re.compile(

                r"\bradial pulse\s+(?:is|was)\s+absent\b",

                re.IGNORECASE,

            ),

        ),

        (

            "mental_status",

            "alert",

            re.compile(

                r"\bmental status\s+(?:is|was)\s+alert\b",

                re.IGNORECASE,

            ),

        ),

        (

            "mental_status",

            "confused",

            re.compile(

                r"\bmental status\s+(?:is|was)\s+confused\b",

                re.IGNORECASE,

            ),

        ),

        (

            "mental_status",

            "unresponsive",

            re.compile(

                r"\bmental status\s+(?:is|was)\s+unresponsive\b",

                re.IGNORECASE,

            ),

        ),

        (

            "ambulatory_status",

            "ambulatory",

            re.compile(

                r"\bambulatory status\s+(?:is|was)\s+ambulatory\b",

                re.IGNORECASE,

            ),

        ),

        (

            "ambulatory_status",

            "non_ambulatory",

            re.compile(

                r"\bambulatory status\s+(?:is|was)\s+"

                r"(?:non[-_ ]ambulatory)\b",

                re.IGNORECASE,

            ),

        ),

        (

            "visible_hemorrhage",

            "no visible major hemorrhage",

            re.compile(

                r"\bno visible major hemorrhage\b",

                re.IGNORECASE,

            ),

        ),

        (

            "visible_hemorrhage",

            "visible major hemorrhage",

            re.compile(

                r"(?<!no )\bvisible major hemorrhage\b",

                re.IGNORECASE,

            ),

        ),

        (

            "follows_commands",

            "does not follow commands",

            re.compile(

                r"\bdoes not follow commands\b",

                re.IGNORECASE,

            ),

        ),

        (

            "follows_commands",

            "follows commands",

            re.compile(

                r"\bfollows commands\b",

                re.IGNORECASE,

            ),

        ),

    )



    # Supports both:

    # "simulation second 0"

    # "simulation seconds 0 and 60"

    _SIMULATION_TIME_PATTERN = re.compile(

        r"\bsimulation seconds?\s+(\d+)"

        r"(?:\s+and\s+(\d+))?\b",

        re.IGNORECASE,

    )



    _PATIENT_ID_PATTERN = re.compile(
        r"\b(P(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9][A-Za-z0-9_-]{0,62})\b",
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

                    max(0, match.start() - 80):match.start()

                ]



                patient_matches = list(

                    cls._PATIENT_ID_PATTERN.finditer(preceding_text)

                )



                if not patient_matches:

                    reasons.append(

                        "Scene narration contains a clinical measurement "

                        "without an identifiable patient."

                    )

                    continue



                patient_id = patient_matches[-1].group(1).casefold()

                approved_value = evidence.get(

                    (patient_id, observation_type)

                )



                if approved_value is None or claimed_value != approved_value:

                    reasons.append(

                        "Scene narration contains a clinical measurement "

                        "that does not match the approved observation "

                        "for that patient."

                    )



        return reasons



    @classmethod

    def _extract_timestamps(cls, text: str) -> list[int]:

        """Extract timestamps from recognized simulation-time expressions."""

        timestamps = []



        for match in cls._SIMULATION_TIME_PATTERN.finditer(text):

            timestamps.extend(

                int(value)

                for value in match.groups()

                if value is not None

            )



        return timestamps



    @classmethod
    def _extract_latest_timestamps(cls, text: str) -> list[int]:
        """Return timestamps from the nearest preceding time expression."""
        matches = list(cls._SIMULATION_TIME_PATTERN.finditer(text))
        if not matches:
            return []
        return [
            int(value)
            for value in matches[-1].groups()
            if value is not None
        ]
    @classmethod
    def _temporal_claim_context(cls, text: str) -> str:
        """Carry forward only an explicitly scoped patient/time declaration."""
        segments = re.split(r"(?<=[.!?])\s+|\r?\n+", text)

        current_segment = segments[-1].strip()

        if len(segments) < 2:
            return current_segment

        previous_segment = segments[-2].strip()

        explicit_patient_scope = re.search(
            r"\bfor\s+patient\s+P(?=[A-Za-z0-9_-]*\d)"
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}\b",
            previous_segment,
            re.IGNORECASE,
        )

        explicit_timestamp = cls._SIMULATION_TIME_PATTERN.search(
            previous_segment
        )

        if explicit_patient_scope and explicit_timestamp:
            return f"{previous_segment} {current_segment}"

        return current_segment

    @classmethod
    def _extract_unambiguous_patient_id(cls, text: str) -> str | None:
        """Return a patient ID only when exactly one unique ID is present."""
        patient_ids = {
            match.group(1).casefold()
            for match in cls._PATIENT_ID_PATTERN.finditer(text)
        }

        if len(patient_ids) != 1:
            return None

        return next(iter(patient_ids))

    @classmethod
    def _review_temporal_measurements(

        cls,

        request: AgentRequest,

        response: AgentResponse,

    ) -> list[str]:
        evidence = {

            (

                observation.patient_id.casefold(),

                observation.observation_type,

                cls._normalize_value(observation.value),

                observation.observed_at_seconds,

            )

            for observation in request.temporal_observations

        }



        reasons = []



        for observation_type, pattern in cls._TEMPORAL_PATTERNS:

            for match in pattern.finditer(response.content):

                preceding_text = cls._temporal_claim_context(
                    response.content[:match.start()]
                )



                patient_id = cls._extract_unambiguous_patient_id(
                    preceding_text
                )

                if patient_id is None:
                    reasons.append(
                        "Temporal narration contains a measurement "
                        "without one unambiguous patient identifier."
                    )
                    continue



                timestamps = cls._extract_latest_timestamps(preceding_text)



                if not timestamps:

                    reasons.append(

                        "Temporal narration contains a measurement "

                        "without an identifiable simulation timestamp."

                    )

                    continue





                claimed_value = cls._normalize_value(match.group(1))



                for timestamp in timestamps:

                    claimed_evidence = (

                        patient_id,

                        observation_type,

                        claimed_value,

                        timestamp,

                    )



                    if claimed_evidence not in evidence:

                        reasons.append(

                            "Temporal narration contains a measurement "

                            "that does not match the approved patient, "

                            "observation type, value, and timestamp."

                        )



        for observation_type, expected_value, pattern in (

            cls._TEMPORAL_CATEGORICAL_PATTERNS

        ):

            for match in pattern.finditer(response.content):

                preceding_text = response.content[:match.start()]



                patient_id = cls._extract_unambiguous_patient_id(
                    preceding_text
                )

                if patient_id is None:
                    reasons.append(
                        "Temporal narration contains a categorical "
                        "observation without one unambiguous patient "
                        "identifier."
                    )
                    continue



                timestamps = cls._extract_latest_timestamps(preceding_text)



                if not timestamps:

                    reasons.append(

                        "Temporal narration contains a categorical "

                        "observation without an identifiable simulation "

                        "timestamp."

                    )

                    continue







                for timestamp in timestamps:

                    claimed_evidence = (

                        patient_id,

                        observation_type,

                        cls._normalize_value(expected_value),

                        timestamp,

                    )



                    if claimed_evidence not in evidence:

                        reasons.append(

                            "Temporal narration contains a categorical "

                            "observation that does not match the approved "

                            "patient, observation type, value, and timestamp."

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



                        if claimed_value not in self._normalize_value(

                            disclosed_text

                        ):

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

            reasons.extend(

                self._review_scene_measurements(request, response)

            )



        elif request.task == AgentTask.TEMPORAL_SCENE_NARRATION:

            reasons.extend(

                self._review_temporal_measurements(request, response)

            )



        return SafetyReview(

            approved=not reasons,

            reasons=tuple(reasons),

        )
