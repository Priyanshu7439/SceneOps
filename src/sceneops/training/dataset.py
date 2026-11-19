from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sceneops.agents.disclosure_rules import (
    DisclosurePacketBuilder,
    PatientDisclosureRegistry,
    PatientQuestion,
)
from sceneops.simulator.scenario_factory import (
    create_mass_casualty_scenario,
)

GENERATOR_VERSION = "2.1.0"

SCENARIO_COUNT = 92
DEFAULT_SCENARIO_INDICES = tuple(range(1, SCENARIO_COUNT + 1))

PAIN_QUESTIONS = (
    "Are you experiencing any pain? How severe is it?",
    "Can you tell me how much pain you are feeling?",
    "On a scale of 0 to 10, how bad is your pain?",
)

PAIN_RESPONSES = (
    "My pain is {value}.",
    "I'd say my pain is around {value}.",
    "It's about {value}.",
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class DatasetExample(BaseModel):
    """One traceable synthetic supervised fine-tuning example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    task: Literal["patient_dialogue"]
    generator_version: str = Field(min_length=1)
    disclosure_fact: str = Field(min_length=1)
    disclosure_value: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...]

    def to_training_record(self) -> dict:
        """Return conversation-only format used by SFT pipelines."""
        return {"messages": [message.model_dump() for message in self.messages]}


class PatientDialogueDatasetGenerator:
    """Generate controlled dialogue examples across multiple scenarios."""

    def generate(
        self,
        scenario_indices: tuple[int, ...] = DEFAULT_SCENARIO_INDICES,
    ) -> list[DatasetExample]:
        if not scenario_indices:
            raise ValueError("At least one scenario index is required.")

        if len(scenario_indices) != len(set(scenario_indices)):
            raise ValueError("Scenario indices must be unique.")

        examples: list[DatasetExample] = []

        for scenario_index in scenario_indices:
            scenario = create_mass_casualty_scenario(scenario_index)
            registry = PatientDisclosureRegistry()

            for patient_index, patient in enumerate(scenario.patients):
                # Controlled synthetic variation, not clinical inference.
                synthetic_pain_level = (patient_index * 3 + scenario_index * 2) % 11

                patient.update_pain(synthetic_pain_level)

                registry.record_answer(
                    patient,
                    PatientQuestion.PAIN_LEVEL,
                )

                packet = DisclosurePacketBuilder.build(
                    patient,
                    disclosure_registry=registry,
                )

                if len(packet.current_disclosures) != 1:
                    raise ValueError(
                        f"Expected exactly one disclosure for "
                        f"{patient.patient_id} in {scenario.scenario_id}."
                    )

                disclosure = packet.current_disclosures[0]

                if disclosure.fact != "reported_pain_level":
                    raise ValueError(f"Unexpected disclosure fact: {disclosure.fact}")

                system_prompt = (
                    "You are roleplaying a patient in a synthetic educational "
                    "mass-casualty simulation. Respond naturally as the patient. "
                    "Use only the information explicitly available in this "
                    "conversation and the authorized disclosure below. "
                    "Do not invent symptoms, measurements, injuries, diagnoses, "
                    "or other medical facts. Do not reveal hidden simulator state. "
                    f"Personality: {packet.profile.personality}. "
                    f"Speech style: {packet.profile.speech_style}. "
                    f"Authorized disclosure: {disclosure.fact} = "
                    f"{disclosure.value}."
                )

                for variation, question in enumerate(PAIN_QUESTIONS):
                    answer = PAIN_RESPONSES[variation].format(value=disclosure.value)

                    examples.append(
                        DatasetExample(
                            example_id=(
                                f"{scenario.scenario_id}-"
                                f"{patient.patient_id}-"
                                f"pain-{variation + 1:02d}"
                            ),
                            scenario_id=scenario.scenario_id,
                            patient_id=patient.patient_id,
                            task="patient_dialogue",
                            generator_version=GENERATOR_VERSION,
                            disclosure_fact=disclosure.fact,
                            disclosure_value=disclosure.value,
                            messages=(
                                ChatMessage(
                                    role="system",
                                    content=system_prompt,
                                ),
                                ChatMessage(
                                    role="user",
                                    content=question,
                                ),
                                ChatMessage(
                                    role="assistant",
                                    content=answer,
                                ),
                            ),
                        )
                    )

        return examples
