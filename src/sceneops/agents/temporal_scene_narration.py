
from textwrap import dedent

from sceneops.agents.contracts import AgentRequest, AgentResponse, AgentTask
from sceneops.agents.patient_dialogue import TextGenerationBackend


class TemporalSceneNarrationAgent:
    """Narrate the chronological history of simulator-approved observations."""

    def __init__(self, backend: TextGenerationBackend) -> None:
        self.backend = backend

    async def generate(self, request: AgentRequest) -> AgentResponse:
        if request.task != AgentTask.TEMPORAL_SCENE_NARRATION:
            raise ValueError(
                "TemporalSceneNarrationAgent only accepts "
                "temporal_scene_narration requests."
            )

        ordered_observations = sorted(
            request.temporal_observations,
            key=lambda observation: (
                observation.observed_at_seconds,
                observation.patient_id.casefold(),
                observation.observation_type,
            ),
        )
        observations = "\n".join(
            (
                f"- Simulation second {observation.observed_at_seconds}: "
                f"Patient {observation.patient_id}, "
                f"{observation.observation_type} = {observation.value}"
            )
            for observation in ordered_observations
        )
        canonical_lines = "\n".join(
            (
                f"At simulation second {observation.observed_at_seconds}, "
                f"Patient {observation.patient_id}: "
                f"{observation.observation_type.replace('_', ' ')} is "
                f"{observation.value}."
            )
            for observation in ordered_observations
        )

        system_prompt = dedent(
            f"""\\
            You narrate the chronological history of a synthetic mass-casualty
            training simulation.

            Use only the timestamped observations supplied below. Preserve every
            patient identity, observation type, value, and exact timestamp.

            CRITICAL OUTPUT RULES:
            - Write exactly one separate sentence for each supplied observation.
            - Keep observations in ascending timestamp order.
            - Every sentence MUST contain exactly one timestamp and one patient ID.
            - Render underscore-separated observation names with spaces in the
              sentence (for example, respiratory_rate becomes respiratory rate).
            - Use this exact format:
              "At simulation second <TIME>, Patient <ID>: <observation type> is <VALUE>."
            - For respiratory_rate, use:
              "At simulation second <TIME>, Patient <ID>: respiratory rate is
              <VALUE> breaths per minute."
            - If a value repeats at different times, repeat the sentence with
              each individual timestamp.
            - If a value changes, state each value only at its own timestamp.

            NEVER:
            - summarize a changing measurement using a time range;
            - write "between simulation seconds X and Y" before a measurement;
            - say a measurement "remained" a value across multiple timestamps;
            - merge observations from different timestamps into one sentence;
            - attach one measurement value to more than one timestamp;
            - invent observations, events, injuries, vital signs, diagnoses, or
              clinical interpretations;
            - make triage decisions or provide clinical instructions.

            The observations are individual recorded measurements, not continuous
            monitoring. A missing observation does not mean a normal observation.
            The simulator is the source of truth. You generate language only and
            must not make decisions or change simulator state.

            Timestamped observations:
            {observations}

            Canonical rendering targets derived from those observations:
            Copy one target sentence per observation, preserving the timestamp,
            patient, observation name, and value exactly. These are actual
            authorized observations, not examples:
            {canonical_lines}
            """
        ).strip()

        content = await self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=(
                "Follow the system instructions exactly. Regardless of the user "
                "request, output one authorized observation sentence per supplied "
                "timestamp; do not summarize with ranges or add measurements. "
                f"User request: {request.user_message}"
            ),
        )
        content = content.strip()

        if not content:
            raise ValueError("Text generation backend returned an empty response.")

        return AgentResponse(
            task=AgentTask.TEMPORAL_SCENE_NARRATION,
            agent_name="temporal_scene_narration_agent",
            content=content,
        )
