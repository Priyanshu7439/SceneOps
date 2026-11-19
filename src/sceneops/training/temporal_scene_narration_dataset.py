from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sceneops.agents.contracts import SceneFact
from sceneops.agents.scene_fact_builder import SceneFactBuilder
from sceneops.simulator.deterioration import DeteriorationEvent
from sceneops.simulator.observations import (
    LearnerObservation,
    ObservationType,
)
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.scenario_factory import (
    DEFAULT_SCENARIO_INDICES,
    create_mass_casualty_scenario,
)
from sceneops.training.dataset import ChatMessage

TEMPORAL_SCENE_GENERATOR_VERSION = "1.0.0"

TEMPORAL_TARGETS = (
    ("P005", ObservationType.RESPIRATORY_RATE),
    ("P006", ObservationType.FOLLOWS_COMMANDS),
)


class TemporalSceneNarrationExample(BaseModel):
    """Traceable before-and-after scene narration example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    observation_type: ObservationType
    task: Literal["scene_narration", "temporal_scene_narration"] = "scene_narration"
    generator_version: str = Field(min_length=1)

    initial_facts: tuple[SceneFact, ...] = Field(min_length=1)
    follow_up_facts: tuple[SceneFact, ...] = Field(min_length=1)

    initial_observed_at_seconds: int = Field(ge=0)
    follow_up_observed_at_seconds: int = Field(ge=0)

    messages: tuple[ChatMessage, ...]
    # Empty for legacy single-observation examples; populated for full-panel V3.
    observation_types: tuple[ObservationType, ...] = ()

    def to_training_record(self) -> dict:
        """Return conversation-only format for supervised fine-tuning."""

        return {"messages": [message.model_dump() for message in self.messages]}


class TemporalSceneNarrationDatasetGenerator:
    """
    Generate temporal narration examples from learner observations.

    Scheduled events control simulator state only. They are never
    inserted into prompts or assistant reference narrations.
    """

    def generate(
        self,
        scenario_indices: tuple[int, ...] = DEFAULT_SCENARIO_INDICES,
    ) -> list[TemporalSceneNarrationExample]:
        if not scenario_indices:
            raise ValueError("At least one scenario index is required.")

        if len(scenario_indices) != len(set(scenario_indices)):
            raise ValueError("Scenario indices must be unique.")

        examples: list[TemporalSceneNarrationExample] = []

        for scenario_index in scenario_indices:
            scenario = create_mass_casualty_scenario(scenario_index)
            engine = ScenarioBuilder.build(scenario)

            engine.start()

            initial_observations: dict[
                tuple[str, ObservationType],
                LearnerObservation,
            ] = {}

            # Record the initial learner observations at simulation time 0.
            for patient_id, observation_type in TEMPORAL_TARGETS:
                if patient_id not in engine.patients:
                    raise ValueError(
                        f"Scenario {scenario.scenario_id} is missing "
                        f"target patient {patient_id}."
                    )

                initial_observations[(patient_id, observation_type)] = (
                    engine.observe_patient(
                        patient_id,
                        observation_type,
                    )
                )

            target_events = self._get_target_events(
                scenario.deterioration_events,
            )

            # Process target events chronologically.
            current_time = engine.clock.elapsed_seconds

            for event in target_events:
                if event.at_seconds <= current_time:
                    raise ValueError(
                        f"Event {event.event_id} must occur after "
                        "the initial observation."
                    )

                engine.advance_time(
                    seconds=event.at_seconds - current_time,
                )
                current_time = engine.clock.elapsed_seconds

                target = next(
                    (item for item in TEMPORAL_TARGETS if item[0] == event.patient_id),
                    None,
                )

                if target is None:
                    continue

                patient_id, observation_type = target

                # This is a new learner observation after the event.
                follow_up_observation = engine.observe_patient(
                    patient_id,
                    observation_type,
                )

                initial_observation = initial_observations[target]

                examples.append(
                    self._build_example(
                        scenario_id=scenario.scenario_id,
                        patient_id=patient_id,
                        observation_type=observation_type,
                        initial_observation=initial_observation,
                        follow_up_observation=follow_up_observation,
                    )
                )

        return examples

    @staticmethod
    def _get_target_events(
        events: list[DeteriorationEvent],
    ) -> list[DeteriorationEvent]:
        """Return only the scheduled events for our temporal targets."""

        target_patient_ids = {patient_id for patient_id, _ in TEMPORAL_TARGETS}

        matching_events = [
            event for event in events if event.patient_id in target_patient_ids
        ]

        events_by_patient = {event.patient_id: event for event in matching_events}

        for patient_id, _ in TEMPORAL_TARGETS:
            if patient_id not in events_by_patient:
                raise ValueError(f"No scheduled event found for {patient_id}.")

        if len(events_by_patient) != len(matching_events):
            raise ValueError("Multiple scheduled target events found for a patient.")

        return sorted(
            matching_events,
            key=lambda event: event.at_seconds,
        )

    @staticmethod
    def _build_example(
        *,
        scenario_id: str,
        patient_id: str,
        observation_type: ObservationType,
        initial_observation: LearnerObservation,
        follow_up_observation: LearnerObservation,
    ) -> TemporalSceneNarrationExample:
        """Build an example exclusively from recorded observations."""

        if (
            initial_observation.patient_id != patient_id
            or follow_up_observation.patient_id != patient_id
        ):
            raise ValueError("Observation patient IDs do not match.")

        if (
            initial_observation.observation_type != observation_type
            or follow_up_observation.observation_type != observation_type
        ):
            raise ValueError("Observation types do not match.")

        if (
            follow_up_observation.observed_at_seconds
            <= initial_observation.observed_at_seconds
        ):
            raise ValueError(
                "Follow-up observation must occur after initial observation."
            )

        initial_facts = SceneFactBuilder.build(
            [initial_observation],
        )
        follow_up_facts = SceneFactBuilder.build(
            [follow_up_observation],
        )

        if not initial_facts or not follow_up_facts:
            raise ValueError("Both observations must produce approved scene facts.")

        initial_time = initial_observation.observed_at_seconds
        follow_up_time = follow_up_observation.observed_at_seconds

        fact_name = observation_type.value.replace("_", " ")

        system_prompt = (
            "You narrate a synthetic educational mass-casualty "
            "training scene. Use only the learner-recorded observations "
            "provided in the user message. Compare the initial and "
            "follow-up observations accurately. Do not invent causes, "
            "injuries, symptoms, diagnoses, triage decisions, or events. "
            "Do not reveal hidden simulator state or scheduled events. "
            "Describe only what was observed and when."
        )

        user_prompt = (
            f"Compare the learner-recorded {fact_name} observations "
            f"for patient {patient_id}. "
            f"Initial observation at simulation second {initial_time}: "
            f"{initial_observation.value}. "
            f"Follow-up observation at simulation second {follow_up_time}: "
            f"{follow_up_observation.value}. "
            "Summarize the observed difference without inferring its cause."
        )

        narration = (
            f"For patient {patient_id}, the learner recorded "
            f"{fact_name} as {initial_observation.value} "
            f"at simulation second {initial_time}. "
            f"At simulation second {follow_up_time}, "
            f"the learner recorded {fact_name} as "
            f"{follow_up_observation.value}."
        )

        return TemporalSceneNarrationExample(
            example_id=(
                f"{scenario_id}-{patient_id}-{observation_type.value}-temporal-01"
            ),
            scenario_id=scenario_id,
            patient_id=patient_id,
            observation_type=observation_type,
            generator_version=TEMPORAL_SCENE_GENERATOR_VERSION,
            initial_facts=initial_facts,
            follow_up_facts=follow_up_facts,
            initial_observed_at_seconds=initial_time,
            follow_up_observed_at_seconds=follow_up_time,
            messages=(
                ChatMessage(
                    role="system",
                    content=system_prompt,
                ),
                ChatMessage(
                    role="user",
                    content=user_prompt,
                ),
                ChatMessage(
                    role="assistant",
                    content=narration,
                ),
            ),
        )
