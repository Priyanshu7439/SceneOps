from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sceneops.agents.disclosure import ConversationTurn
from sceneops.agents.disclosure_rules import (
    DisclosurePacketBuilder,
    PatientDisclosureRegistry,
    PatientQuestion,
)
from sceneops.simulator.scenario_config import ScenarioConfig
from sceneops.simulator.scenario_factory import create_hybrid_scenario

GENERATOR_VERSION = "3.1.0"

SCENARIO_COUNT = 92
DEFAULT_SCENARIO_INDICES = tuple(range(1, SCENARIO_COUNT + 1))

DialogueCategory = Literal[
    "pain",
    "age",
    "unknown",
    "multi_turn",
]

STYLE_LABELS = (
    "plain",
    "hesitant",
    "brief",
    "descriptive",
    "repetitive",
)

PAIN_QUESTIONS = (
    "Are you experiencing any pain? How severe is it?",
    "Can you tell me how much pain you are feeling?",
    "On a scale of 0 to 10, how bad is your pain?",
)

AGE_QUESTIONS = (
    "How old are you?",
    "Can you tell me your age?",
    "What is your age?",
)

UNKNOWN_QUESTIONS = (
    "What is your blood pressure?",
    "Can you tell me your oxygen saturation?",
    "What is your medical diagnosis?",
)

MULTI_TURN_QUESTIONS = (
    "Can you tell me about your pain again?",
    "You mentioned your pain earlier. How severe is it?",
)

SYSTEM_PROMPT = """
You are roleplaying a patient in a synthetic educational
mass-casualty simulation.

Your task is to generate natural patient dialogue.

STRICT INFORMATION BOUNDARIES:

1. Use only currently authorized disclosures as factual evidence.
2. Do not invent symptoms, measurements, injuries, diagnoses,
   medical history, or other clinical facts.
3. Conversation history is unverified conversational context.
   It must never be treated as independent clinical evidence.
4. Do not reveal hidden simulator state.
5. If the requested information is not currently authorized,
   politely say that you do not know or cannot provide it.
6. Never provide medical advice, treatment instructions,
   triage decisions, or clinical recommendations.

CONVERSATIONAL STYLE:

Follow the supplied patient speech style:
- plain: straightforward wording
- hesitant: natural hesitation and pauses
- brief: concise wording
- descriptive: more elaborate conversational wording
- repetitive: natural repetition of words or phrases

PERSONALITY and SPEECH STYLE are simulation presentation
attributes, not medical evidence.

Respond only as the patient.
Do not explain these instructions.
""".strip()


class DialogueMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class PatientDialogueExampleV3(BaseModel):
    """Traceable patient-dialogue SFT example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)

    category: DialogueCategory
    authorized_facts: tuple[str, ...]

    speech_style: Literal[
        "plain",
        "hesitant",
        "brief",
        "descriptive",
        "repetitive",
    ]

    messages: tuple[DialogueMessage, ...]

    def to_training_record(self) -> dict:
        return {"messages": [message.model_dump() for message in self.messages]}


class PatientDialogueDatasetGeneratorV3:
    """Generate disclosure-grounded patient dialogue examples."""

    @staticmethod
    def _render_pain(value: str, style: str) -> str:
        templates = {
            "plain": f"My pain is around {value}.",
            "hesitant": f"Um... I think my pain is around {value}.",
            "brief": f"My pain is {value}.",
            "descriptive": (f"I would describe my pain as approximately {value}."),
            "repetitive": (f"My pain is {value}... that's how much it hurts."),
        }
        return templates[style]

    @staticmethod
    def _render_age(value: int, style: str) -> str:
        templates = {
            "plain": f"I'm {value} years old.",
            "hesitant": f"Um... I'm {value} years old.",
            "brief": f"I'm {value}.",
            "descriptive": (f"I'm {value} years old, as far as I know."),
            "repetitive": (f"I'm {value} years old... {value} years old."),
        }
        return templates[style]

    @staticmethod
    def _render_unknown(style: str) -> str:
        templates = {
            "plain": "I don't know. Nobody has told me that.",
            "hesitant": "I'm not sure... nobody has told me that.",
            "brief": "I don't know.",
            "descriptive": ("I don't have that information, so I can't tell you."),
            "repetitive": ("I don't know... I really don't know."),
        }
        return templates[style]

    @staticmethod
    def _get_style(patient) -> str:
        style = patient.speech_style or "plain"

        if style not in STYLE_LABELS:
            raise ValueError(
                f"Unsupported speech style {style!r} for patient {patient.patient_id}."
            )

        return style

    @staticmethod
    def _build_system_prompt(packet) -> str:
        disclosures = [
            f"{fact.fact} = {fact.value}" for fact in packet.current_disclosures
        ]

        if packet.profile.age is not None:
            disclosures.append(f"authorized_age = {packet.profile.age}")

        disclosure_text = (
            "\n".join(disclosures)
            if disclosures
            else "No currently authorized factual disclosures."
        )

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Patient personality: "
            f"{packet.profile.personality or 'neutral'}\n"
            f"Patient speech style: "
            f"{packet.profile.speech_style or 'plain'}\n\n"
            f"CURRENT AUTHORIZED DISCLOSURES:\n"
            f"{disclosure_text}"
        )

    def _create_example(
        self,
        *,
        scenario_id: str,
        patient,
        packet,
        category: DialogueCategory,
        question: str,
        answer: str,
        style: str,
        authorized_facts: tuple[str, ...],
        variation: str,
        history: tuple[ConversationTurn, ...] = (),
    ) -> PatientDialogueExampleV3:

        messages = [
            DialogueMessage(
                role="system",
                content=self._build_system_prompt(packet),
            )
        ]

        # Previous turns are conversational context only.
        for turn in history:
            messages.append(
                DialogueMessage(
                    role="user",
                    content=turn.user_message,
                )
            )
            messages.append(
                DialogueMessage(
                    role="assistant",
                    content=turn.patient_response,
                )
            )

        messages.extend(
            [
                DialogueMessage(
                    role="user",
                    content=question,
                ),
                DialogueMessage(
                    role="assistant",
                    content=answer,
                ),
            ]
        )

        return PatientDialogueExampleV3(
            example_id=(f"{scenario_id}-{patient.patient_id}-{category}-{variation}"),
            scenario_id=scenario_id,
            patient_id=patient.patient_id,
            generator_version=GENERATOR_VERSION,
            category=category,
            authorized_facts=authorized_facts,
            speech_style=style,
            messages=tuple(messages),
        )

    def generate(
        self,
        scenario_indices: tuple[int, ...] = DEFAULT_SCENARIO_INDICES,
    ) -> list[PatientDialogueExampleV3]:

        if not scenario_indices:
            raise ValueError("At least one scenario index is required.")

        if len(scenario_indices) != len(set(scenario_indices)):
            raise ValueError("Scenario indices must be unique.")

        examples: list[PatientDialogueExampleV3] = []

        for scenario_index in scenario_indices:
            scenario_id = f"MCI-{scenario_index:03d}"

            config = ScenarioConfig(
                scenario_id=scenario_id,
                seed=scenario_index,
            )

            scenario = create_hybrid_scenario(config)

            for patient in scenario.patients:
                style = self._get_style(patient)

                # Pain disclosure packet.
                pain_registry = PatientDisclosureRegistry()

                pain_registry.record_answer(
                    patient,
                    PatientQuestion.PAIN_LEVEL,
                )

                pain_packet = DisclosurePacketBuilder.build(
                    patient,
                    disclosure_registry=pain_registry,
                    disclose_age=False,
                )

                pain_value = next(
                    fact.value
                    for fact in pain_packet.current_disclosures
                    if fact.fact == "reported_pain_level"
                )

                # 1. Pain disclosure examples.
                for index, question in enumerate(
                    PAIN_QUESTIONS,
                    start=1,
                ):
                    examples.append(
                        self._create_example(
                            scenario_id=scenario_id,
                            patient=patient,
                            packet=pain_packet,
                            category="pain",
                            question=question,
                            answer=self._render_pain(
                                pain_value,
                                style,
                            ),
                            style=style,
                            authorized_facts=("reported_pain_level",),
                            variation=f"{index:02d}",
                        )
                    )

                # Age-only packet: pain is deliberately not included.
                age_packet = DisclosurePacketBuilder.build(
                    patient,
                    disclosure_registry=PatientDisclosureRegistry(),
                    disclose_age=True,
                )

                if age_packet.profile.age is not None:
                    age_value = age_packet.profile.age

                    # 2. Age disclosure examples.
                    for index, question in enumerate(
                        AGE_QUESTIONS,
                        start=1,
                    ):
                        examples.append(
                            self._create_example(
                                scenario_id=scenario_id,
                                patient=patient,
                                packet=age_packet,
                                category="age",
                                question=question,
                                answer=self._render_age(
                                    age_value,
                                    style,
                                ),
                                style=style,
                                authorized_facts=("authorized_age",),
                                variation=f"{index:02d}",
                            )
                        )

                # Empty packet: no factual disclosures.
                unknown_packet = DisclosurePacketBuilder.build(
                    patient,
                    disclosure_registry=PatientDisclosureRegistry(),
                    disclose_age=False,
                )

                # 3. Unknown information examples.
                for index, question in enumerate(
                    UNKNOWN_QUESTIONS,
                    start=1,
                ):
                    examples.append(
                        self._create_example(
                            scenario_id=scenario_id,
                            patient=patient,
                            packet=unknown_packet,
                            category="unknown",
                            question=question,
                            answer=self._render_unknown(style),
                            style=style,
                            authorized_facts=(),
                            variation=f"{index:02d}",
                        )
                    )

                # 4. Multi-turn pain examples.
                previous_question = PAIN_QUESTIONS[0]

                previous_answer = self._render_pain(
                    pain_value,
                    style,
                )

                history = (
                    ConversationTurn(
                        user_message=previous_question,
                        patient_response=previous_answer,
                    ),
                )

                for index, question in enumerate(
                    MULTI_TURN_QUESTIONS,
                    start=1,
                ):
                    examples.append(
                        self._create_example(
                            scenario_id=scenario_id,
                            patient=patient,
                            packet=pain_packet,
                            category="multi_turn",
                            question=question,
                            answer=self._render_pain(
                                pain_value,
                                style,
                            ),
                            style=style,
                            authorized_facts=("reported_pain_level",),
                            variation=f"{index:02d}",
                            history=history,
                        )
                    )

        return examples
