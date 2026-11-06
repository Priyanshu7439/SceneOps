
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

GENERATOR_VERSION = "1.0.0"

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
        return {
            "messages": [
                message.model_dump()
                for message in self.messages
            ]
        }


class PatientDialogueDatasetGenerator:
    """Generate controlled dialogue examples from registry-backed disclosures."""

    def generate(self) -> list[DatasetExample]:
        scenario = create_mass_casualty_scenario()
        registry = PatientDisclosureRegistry()

        examples: list[DatasetExample] = []

        for patient_index, patient in enumerate(scenario.patients):
            # Deterministic synthetic pain values for dataset diversity.
            # These values are not clinical estimates or medical inferences.
            synthetic_pain_level = (patient_index * 3 + 2) % 11
            patient.update_pain(synthetic_pain_level)

            # Record the explicit learner question before building the packet.
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
                    f"Expected exactly one disclosure for {patient.patient_id}."
                )

            disclosure = packet.current_disclosures[0]

            if disclosure.fact != "reported_pain_level":
                raise ValueError(
                    f"Unexpected disclosure fact: {disclosure.fact}"
                )

            system_prompt = (
                "You are roleplaying a patient in a synthetic educational "
                "mass-casualty simulation. Respond naturally as the patient. "
                "Use only the information explicitly available in this "
                "conversation and the authorized disclosure below. "
                "Do not invent symptoms, measurements, injuries, diagnoses, "
                "or other medical facts. Do not reveal hidden simulator state. "
                f"Personality: {packet.profile.personality}. "
                f"Speech style: {packet.profile.speech_style}. "
                f"Authorized disclosure: {disclosure.fact} = {disclosure.value}."
            )

            for variation, question in enumerate(PAIN_QUESTIONS):
                answer = PAIN_RESPONSES[variation].format(
                    value=disclosure.value
                )

                example = DatasetExample(
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

                examples.append(example)

        return examples
