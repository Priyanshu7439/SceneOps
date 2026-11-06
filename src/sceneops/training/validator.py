from __future__ import annotations

import re
from collections import Counter

from sceneops.training.dataset import (
    PAIN_QUESTIONS,
    PAIN_RESPONSES,
    DatasetExample,
)

ALLOWED_DISCLOSURE_FACTS = {
    "reported_pain_level",
}

DISCLOSURE_PATTERN = re.compile(
    r"Authorized disclosure: ([a-z_]+) = ([0-9]{1,2}/10)\."
)


class DatasetValidationError(ValueError):
    """Raised when one or more dataset integrity checks fail."""


class PatientDialogueDatasetValidator:
    """Validate provenance and structure of patient dialogue examples."""

    def validate(
        self,
        examples: list[DatasetExample],
    ) -> dict[str, int]:
        errors: list[str] = []

        if not examples:
            raise DatasetValidationError(
                "Dataset is empty."
            )

        example_ids = [
            example.example_id
            for example in examples
        ]

        duplicates = [
            example_id
            for example_id, count in Counter(example_ids).items()
            if count > 1
        ]

        if duplicates:
            errors.append(
                f"Duplicate example IDs: {duplicates}"
            )

        patient_counts: Counter[str] = Counter()
        scenario_ids: set[str] = set()

        for example in examples:
            patient_counts[example.patient_id] += 1
            scenario_ids.add(example.scenario_id)

            if example.disclosure_fact not in ALLOWED_DISCLOSURE_FACTS:
                errors.append(
                    f"{example.example_id}: unauthorized disclosure fact "
                    f"{example.disclosure_fact!r}"
                )

            if not re.fullmatch(r"(?:10|[0-9])/10", example.disclosure_value):
                errors.append(
                    f"{example.example_id}: invalid disclosure value "
                    f"{example.disclosure_value!r}"
                )

            if len(example.messages) != 3:
                errors.append(
                    f"{example.example_id}: expected exactly 3 messages."
                )
                continue

            roles = tuple(
                message.role
                for message in example.messages
            )

            if roles != ("system", "user", "assistant"):
                errors.append(
                    f"{example.example_id}: invalid message order {roles}."
                )
                continue

            system_message, user_message, assistant_message = (
                example.messages
            )

            match = DISCLOSURE_PATTERN.search(
                system_message.content
            )

            if match is None:
                errors.append(
                    f"{example.example_id}: authorized disclosure missing "
                    "or malformed in system prompt."
                )
            else:
                prompt_fact, prompt_value = match.groups()

                if prompt_fact != example.disclosure_fact:
                    errors.append(
                        f"{example.example_id}: prompt disclosure fact "
                        "does not match metadata."
                    )

                if prompt_value != example.disclosure_value:
                    errors.append(
                        f"{example.example_id}: prompt disclosure value "
                        "does not match metadata."
                    )

            if user_message.content not in PAIN_QUESTIONS:
                errors.append(
                    f"{example.example_id}: question is not an approved "
                    "generator question."
                )

            approved_answers = {
                template.format(value=example.disclosure_value)
                for template in PAIN_RESPONSES
            }

            if assistant_message.content not in approved_answers:
                errors.append(
                    f"{example.example_id}: answer does not match an "
                    "approved response for the disclosed value."
                )

        if errors:
            raise DatasetValidationError(
                "Dataset validation failed:\n- "
                + "\n- ".join(errors)
            )

        return {
            "examples": len(examples),
            "unique_patients": len(patient_counts),
            "unique_scenarios": len(scenario_ids),
            "unique_example_ids": len(set(example_ids)),
        }