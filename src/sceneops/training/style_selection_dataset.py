from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sceneops.simulator.scenario_config import ScenarioConfig
from sceneops.simulator.scenario_factory import create_hybrid_scenario

GENERATOR_VERSION = "3.0.0"

SCENARIO_COUNT = 92
EXAMPLES_PER_PATIENT = 3

STYLE_LABELS = (
    "plain",
    "brief",
    "hesitant",
    "descriptive",
    "repetitive",
)

PERSONALITY_STYLE_MAP = {
    "calm": "plain",
    "anxious": "hesitant",
    "cooperative": "descriptive",
    "irritable": "brief",
    "confused": "repetitive",
}

LEARNER_MESSAGES = (
    "How are you feeling right now?",
    "Can you tell me about yourself?",
    "Please describe how you are feeling.",
    "Are you feeling okay?",
    "Can you explain what happened?",
    "I want to understand how you are doing.",
    "Could you tell me a little more?",
    "What would you like me to know?",
    "Can you answer a few questions for me?",
    "Tell me what you are experiencing.",
    "Could you explain that in your own words?",
    "I am listening. Please continue.",
    "Can you tell me what is bothering you?",
    "Would you mind describing your situation?",
    "I did not understand that. Could you explain again?",
    "Can you repeat what you just said?",
    "Is there anything else you want to tell me?",
    "Please tell me a bit more about yourself.",
    "Can you help me understand your response?",
    "Take your time and tell me what you can.",
)


STYLE_SYSTEM_PROMPT = """
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

Follow this synthetic simulation mapping:
- calm personality -> plain
- anxious personality -> hesitant
- cooperative personality -> descriptive
- irritable personality -> brief
- confused personality -> repetitive

Use the patient's personality to select the style.
The learner message provides conversational context but does not
change the personality-to-style mapping.

Do not generate dialogue.
Do not generate symptoms, medical facts, measurements, or diagnoses.
Do not include explanations or additional JSON fields.
""".strip()


class StyleChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class StyleSelectionExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    target_style: Literal[
        "plain",
        "brief",
        "hesitant",
        "descriptive",
        "repetitive",
    ]
    messages: tuple[StyleChatMessage, ...]

    def to_training_record(self) -> dict:
        return {"messages": [message.model_dump() for message in self.messages]}


class StyleSelectionDatasetGenerator:
    """Generate deterministic style-selection SFT examples."""

    def generate(self) -> list[StyleSelectionExample]:
        examples: list[StyleSelectionExample] = []

        for scenario_index in range(1, SCENARIO_COUNT + 1):
            scenario_id = f"STYLE-{scenario_index:03d}"

            config = ScenarioConfig(
                scenario_id=scenario_id,
                seed=scenario_index,
            )

            scenario = create_hybrid_scenario(config)

            for patient_index, patient in enumerate(scenario.patients):
                personality = patient.personality or "neutral"
                style = PERSONALITY_STYLE_MAP.get(personality)

                if style is None:
                    raise ValueError(
                        f"Unsupported personality {personality!r} "
                        f"for patient {patient.patient_id}. "
                        f"Expected one of: "
                        f"{tuple(PERSONALITY_STYLE_MAP)}"
                    )

                # Select three messages deterministically.
                # This preserves reproducibility without relying on
                # Python's randomized hash implementation.
                start_index = (scenario_index + patient_index) % len(LEARNER_MESSAGES)

                selected_messages = tuple(
                    LEARNER_MESSAGES[(start_index + offset) % len(LEARNER_MESSAGES)]
                    for offset in range(EXAMPLES_PER_PATIENT)
                )

                for variation, learner_message in enumerate(
                    selected_messages,
                    start=1,
                ):
                    user_prompt = (
                        f"Personality: {personality}\n"
                        f"Learner message: {learner_message}"
                    )

                    target = json.dumps(
                        {"style": style},
                        ensure_ascii=False,
                    )

                    examples.append(
                        StyleSelectionExample(
                            example_id=(
                                f"{scenario_id}-"
                                f"{patient.patient_id}-"
                                f"style-{variation:02d}"
                            ),
                            scenario_id=scenario_id,
                            patient_id=patient.patient_id,
                            generator_version=GENERATOR_VERSION,
                            target_style=style,
                            messages=(
                                StyleChatMessage(
                                    role="system",
                                    content=STYLE_SYSTEM_PROMPT,
                                ),
                                StyleChatMessage(
                                    role="user",
                                    content=user_prompt,
                                ),
                                StyleChatMessage(
                                    role="assistant",
                                    content=target,
                                ),
                            ),
                        )
                    )

        return examples
