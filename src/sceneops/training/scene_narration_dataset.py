from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sceneops.agents.contracts import SceneFact
from sceneops.agents.scene_fact_builder import SceneFactBuilder
from sceneops.simulator.observations import ObservationType
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.scenario_factory import (
    DEFAULT_SCENARIO_INDICES,
    create_mass_casualty_scenario,
)
from sceneops.training.dataset import ChatMessage

SCENE_GENERATOR_VERSION = "1.0.0"

OBSERVATION_TYPES = (
    ObservationType.RESPIRATORY_RATE,
    ObservationType.RADIAL_PULSE,
    ObservationType.CAPILLARY_REFILL,
    ObservationType.MENTAL_STATUS,
    ObservationType.AMBULATORY_STATUS,
)


class SceneNarrationExample(BaseModel):
    """Traceable synthetic scene narration training example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    task: Literal["scene_narration"] = "scene_narration"
    generator_version: str = Field(min_length=1)
    source_facts: tuple[SceneFact, ...] = Field(min_length=1)
    messages: tuple[ChatMessage, ...]

    def to_training_record(self) -> dict:
        """Return conversation-only format for SFT pipelines."""
        return {"messages": [message.model_dump() for message in self.messages]}


class SceneNarrationDatasetGenerator:
    """Generate narration examples from learner-approved observations."""

    def generate(
        self,
        scenario_indices: tuple[int, ...] = DEFAULT_SCENARIO_INDICES,
    ) -> list[SceneNarrationExample]:
        if not scenario_indices:
            raise ValueError("At least one scenario index is required.")

        if len(scenario_indices) != len(set(scenario_indices)):
            raise ValueError("Scenario indices must be unique.")

        examples: list[SceneNarrationExample] = []

        for scenario_index in scenario_indices:
            scenario = create_mass_casualty_scenario(scenario_index)
            engine = ScenarioBuilder.build(scenario)

            for patient_id in engine.patients:
                # Simulate explicit learner assessment actions.
                for observation_type in OBSERVATION_TYPES:
                    try:
                        engine.observe_patient(
                            patient_id,
                            observation_type,
                        )
                    except ValueError:
                        # Missing values and unsupported observation
                        # workflows are not silently invented.
                        continue

                observations = engine.observation_registry.get_for_patient(patient_id)

                facts = SceneFactBuilder.build(observations)

                if not facts:
                    continue

                patient_facts = [
                    fact.fact.rsplit(":", maxsplit=1)[-1].replace("_", " ")
                    + ": "
                    + fact.value.split(" (observed at simulation second ")[0]
                    for fact in facts
                ]

                observation_time = observations[0].observed_at_seconds

                system_prompt = (
                    "You narrate a synthetic educational mass-casualty "
                    "training scene. Use only the approved observations "
                    "provided. Do not invent injuries, symptoms, diagnoses, "
                    "triage decisions, resources, or events. Do not reveal "
                    "hidden simulator state. Describe observations neutrally."
                )

                user_prompt = (
                    f"Summarize the learner-recorded observations for "
                    f"patient {patient_id} at simulation second "
                    f"{observation_time}."
                )

                # Controlled reference narration, not an LLM-generated label.
                narration = (
                    f"At simulation second {observation_time}, "
                    f"the learner recorded the following observations "
                    f"for patient {patient_id}: " + "; ".join(patient_facts) + "."
                )

                examples.append(
                    SceneNarrationExample(
                        example_id=(f"{scenario.scenario_id}-{patient_id}-scene-01"),
                        scenario_id=scenario.scenario_id,
                        patient_id=patient_id,
                        generator_version=SCENE_GENERATOR_VERSION,
                        source_facts=facts,
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
                )

        return examples
