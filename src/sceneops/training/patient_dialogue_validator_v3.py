from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sceneops.training.patient_dialogue_dataset_v3 import (
    AGE_QUESTIONS,
    MULTI_TURN_QUESTIONS,
    PAIN_QUESTIONS,
    STYLE_LABELS,
    UNKNOWN_QUESTIONS,
    PatientDialogueExampleV3,
)

EXPECTED_FACTS = {
    "pain": ("reported_pain_level",),
    "age": ("authorized_age",),
    "unknown": (),
    "multi_turn": ("reported_pain_level",),
}

EXPECTED_ROLES = {
    "pain": ("system", "user", "assistant"),
    "age": ("system", "user", "assistant"),
    "unknown": ("system", "user", "assistant"),
    "multi_turn": (
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ),
}

EXPECTED_QUESTIONS = {
    "pain": set(PAIN_QUESTIONS),
    "age": set(AGE_QUESTIONS),
    "unknown": set(UNKNOWN_QUESTIONS),
    "multi_turn": set(MULTI_TURN_QUESTIONS),
}

UNKNOWN_RESPONSES = {
    "plain": "I don't know. Nobody has told me that.",
    "hesitant": "I'm not sure... nobody has told me that.",
    "brief": "I don't know.",
    "descriptive": ("I don't have that information, so I can't tell you."),
    "repetitive": "I don't know... I really don't know.",
}


@dataclass
class ValidationReport:
    total_examples: int = 0
    valid_examples: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    style_counts: dict[str, int] = field(default_factory=dict)
    scenario_count: int = 0
    patient_count: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "scenario_count": self.scenario_count,
            "patient_count": self.patient_count,
            "category_counts": self.category_counts,
            "style_counts": self.style_counts,
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"V3 dataset validation failed:\n{details}")


class PatientDialogueDatasetValidatorV3:
    """Validate V3 dialogue examples and disclosure boundaries."""

    @staticmethod
    def _authorized_prompt_facts(system_prompt: str) -> dict[str, str]:
        marker = "CURRENT AUTHORIZED DISCLOSURES:"
        if marker not in system_prompt:
            return {}

        disclosure_section = system_prompt.split(marker, 1)[1]
        facts: dict[str, str] = {}

        for line in disclosure_section.splitlines():
            match = re.fullmatch(
                r"\s*([a-z_]+)\s*=\s*(.+?)\s*",
                line,
            )
            if match:
                facts[match.group(1)] = match.group(2)

        return facts

    @staticmethod
    def _validate_example(
        example: PatientDialogueExampleV3,
        errors: list[str],
    ) -> None:
        prefix = example.example_id

        if example.category not in EXPECTED_FACTS:
            errors.append(f"{prefix}: unsupported category {example.category!r}.")
            return

        if example.speech_style not in STYLE_LABELS:
            errors.append(
                f"{prefix}: unsupported speech style {example.speech_style!r}."
            )

        messages = example.messages
        roles = tuple(message.role for message in messages)

        if roles != EXPECTED_ROLES[example.category]:
            errors.append(f"{prefix}: invalid message role sequence {roles}.")
            return

        system_prompt = messages[0].content
        current_question = messages[-2].content
        current_answer = messages[-1].content

        if current_question not in EXPECTED_QUESTIONS[example.category]:
            errors.append(
                f"{prefix}: question does not match category {example.category!r}."
            )

        expected_facts = EXPECTED_FACTS[example.category]

        if tuple(example.authorized_facts) != expected_facts:
            errors.append(
                f"{prefix}: metadata facts "
                f"{example.authorized_facts!r} do not match "
                f"expected facts {expected_facts!r}."
            )

        prompt_facts = PatientDialogueDatasetValidatorV3._authorized_prompt_facts(
            system_prompt
        )

        if set(prompt_facts) != set(expected_facts):
            errors.append(
                f"{prefix}: system prompt disclosure keys "
                f"{tuple(sorted(prompt_facts))!r} do not match "
                f"expected keys {expected_facts!r}."
            )

        # The style and personality must be represented in the prompt.
        if f"Patient speech style: {example.speech_style}" not in system_prompt:
            errors.append(
                f"{prefix}: speech style is missing or inconsistent "
                "in the system prompt."
            )

        if "Patient personality:" not in system_prompt:
            errors.append(
                f"{prefix}: patient personality is missing from the system prompt."
            )

        if example.category == "pain":
            value = prompt_facts.get("reported_pain_level", "")

            if not re.fullmatch(r"(?:10|[0-9])/10", value):
                errors.append(f"{prefix}: invalid authorized pain value {value!r}.")
            elif value not in current_answer:
                errors.append(
                    f"{prefix}: answer does not contain the authorized pain value."
                )

        elif example.category == "age":
            value = prompt_facts.get("authorized_age", "")

            if not value.isdigit():
                errors.append(f"{prefix}: invalid authorized age {value!r}.")
            elif not re.search(
                rf"\b{re.escape(value)}\b",
                current_answer,
            ):
                errors.append(f"{prefix}: answer does not contain the authorized age.")

        elif example.category == "unknown":
            expected_answer = UNKNOWN_RESPONSES.get(example.speech_style)

            if current_answer != expected_answer:
                errors.append(
                    f"{prefix}: unknown-information answer is "
                    "not the expected safe response for its style."
                )

            if re.search(
                r"\d|mmhg|spo2|oxygen saturation|"
                r"blood pressure|diagnos|fracture|"
                r"internal bleeding|heart rate",
                current_answer,
                flags=re.IGNORECASE,
            ):
                errors.append(
                    f"{prefix}: unknown-information answer contains "
                    "a possible unsupported clinical claim."
                )

        elif example.category == "multi_turn":
            pain_value = prompt_facts.get(
                "reported_pain_level",
                "",
            )

            previous_question = messages[-4].content
            previous_answer = messages[-3].content

            if previous_question not in PAIN_QUESTIONS:
                errors.append(
                    f"{prefix}: invalid previous question in multi-turn history."
                )

            if pain_value not in previous_answer:
                errors.append(
                    f"{prefix}: previous answer does not match "
                    "the currently authorized pain value."
                )

            if pain_value not in current_answer:
                errors.append(
                    f"{prefix}: final answer does not contain "
                    "the authorized pain value."
                )

    def validate(
        self,
        examples: list[PatientDialogueExampleV3],
    ) -> ValidationReport:
        report = ValidationReport(total_examples=len(examples))

        if not examples:
            report.errors.append("Dataset is empty.")
            return report

        category_counts = Counter()
        style_counts = Counter()
        example_ids = set()
        scenario_ids = set()
        patient_keys = set()
        patient_category_counts = defaultdict(Counter)

        for example in examples:
            if example.example_id in example_ids:
                report.errors.append(f"Duplicate example ID: {example.example_id}")
            else:
                example_ids.add(example.example_id)

            category_counts[example.category] += 1
            style_counts[example.speech_style] += 1

            scenario_ids.add(example.scenario_id)
            patient_keys.add((example.scenario_id, example.patient_id))

            patient_category_counts[(example.scenario_id, example.patient_id)][
                example.category
            ] += 1

            self._validate_example(
                example,
                report.errors,
            )

        # Every patient should have the expected category distribution.
        expected_category_counts = {
            "pain": len(PAIN_QUESTIONS),
            "age": len(AGE_QUESTIONS),
            "unknown": len(UNKNOWN_QUESTIONS),
            "multi_turn": len(MULTI_TURN_QUESTIONS),
        }

        for patient_key, counts in patient_category_counts.items():
            for category, expected_count in expected_category_counts.items():
                actual_count = counts.get(category, 0)

                if actual_count != expected_count:
                    report.errors.append(
                        f"{patient_key}: expected {expected_count} "
                        f"{category!r} examples, found {actual_count}."
                    )

        missing_styles = set(STYLE_LABELS) - set(style_counts)

        if missing_styles:
            report.warnings.append(
                "Styles not represented in this dataset: "
                + ", ".join(sorted(missing_styles))
            )

        report.valid_examples = report.total_examples - len(
            {error.split(":")[0] for error in report.errors}
        )

        report.category_counts = dict(category_counts)
        report.style_counts = dict(style_counts)
        report.scenario_count = len(scenario_ids)
        report.patient_count = len(patient_keys)

        return report
