from __future__ import annotations

import re

from sceneops.training.temporal_scene_narration_dataset import (
    TemporalSceneNarrationExample,
)


class TemporalDatasetValidationError(ValueError):
    """Raised when temporal narration data violates dataset rules."""


class TemporalSceneNarrationValidator:
    """Validate temporal examples, source facts and information boundaries."""

    FORBIDDEN_PATTERNS = (
        re.compile(r"\bEV\d+\b", re.IGNORECASE),
        re.compile(r"\bpatient_state_changed\b", re.IGNORECASE),
        re.compile(r"\bpatient_reassessed\b", re.IGNORECASE),
        re.compile(r"\bscheduled deterioration event\b", re.IGNORECASE),
    )

    EXPECTED_ROLES = ("system", "user", "assistant")

    @classmethod
    def validate(
        cls,
        examples: list[TemporalSceneNarrationExample],
    ) -> dict[str, int]:
        if not examples:
            raise TemporalDatasetValidationError("Temporal dataset cannot be empty.")

        example_ids = [example.example_id for example in examples]

        if len(example_ids) != len(set(example_ids)):
            raise TemporalDatasetValidationError(
                "Duplicate temporal example IDs found."
            )

        for example in examples:
            cls._validate_example(example)

        return {
            "examples": len(examples),
            "unique_scenarios": len({example.scenario_id for example in examples}),
            "unique_patients": len({example.patient_id for example in examples}),
            "unique_example_ids": len(set(example_ids)),
        }

    @classmethod
    def _validate_example(
        cls,
        example: TemporalSceneNarrationExample,
    ) -> None:
        if example.follow_up_observed_at_seconds <= example.initial_observed_at_seconds:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: follow-up timestamp must "
                "be later than initial timestamp."
            )

        cls._validate_facts(example)
        cls._validate_messages(example)
        cls._validate_no_hidden_state(example)

    @classmethod
    def _validate_facts(
        cls,
        example: TemporalSceneNarrationExample,
    ) -> None:
        expected_types = example.observation_types or (example.observation_type,)
        if (
            example.observation_types
            and example.observation_type not in example.observation_types
        ):
            raise TemporalDatasetValidationError(
                f"{example.example_id}: primary observation type is not in the panel."
            )
        expected_type_values = {item.value for item in expected_types}

        if len(expected_type_values) != len(expected_types):
            raise TemporalDatasetValidationError(
                f"{example.example_id}: duplicate observation types."
            )

        observed_type_sets = []
        for facts, expected_time, label in (
            (
                example.initial_facts,
                example.initial_observed_at_seconds,
                "initial",
            ),
            (
                example.follow_up_facts,
                example.follow_up_observed_at_seconds,
                "follow-up",
            ),
        ):
            if not facts:
                raise TemporalDatasetValidationError(
                    f"{example.example_id}: {label} facts are empty."
                )

            actual_types = set()
            for fact in facts:
                prefix = f"patient_observation:{example.patient_id}:"
                if not fact.fact.startswith(prefix):
                    raise TemporalDatasetValidationError(
                        f"{example.example_id}: invalid {label} source fact identity."
                    )

                fact_type = fact.fact.removeprefix(prefix)
                if fact_type not in expected_type_values:
                    raise TemporalDatasetValidationError(
                        f"{example.example_id}: unexpected {label} observation type."
                    )
                if fact_type in actual_types:
                    raise TemporalDatasetValidationError(
                        f"{example.example_id}: duplicate {label} observation fact."
                    )
                actual_types.add(fact_type)

                expected_timestamp = f"observed at simulation second {expected_time}"
                if expected_timestamp not in fact.value:
                    raise TemporalDatasetValidationError(
                        f"{example.example_id}: {label} fact timestamp "
                        "does not match example metadata."
                    )

            observed_type_sets.append(actual_types)

        if observed_type_sets[0] != observed_type_sets[1]:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: initial and follow-up observation sets differ."
            )

    @classmethod
    def _validate_messages(
        cls,
        example: TemporalSceneNarrationExample,
    ) -> None:
        if len(example.messages) != 3:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: expected exactly three messages."
            )

        roles = tuple(message.role for message in example.messages)

        if roles != cls.EXPECTED_ROLES:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: invalid message role order."
            )

        user_message = example.messages[1].content
        assistant_message = example.messages[2].content

        for fact in (
            *example.initial_facts,
            *example.follow_up_facts,
        ):
            observed_value = fact.value.split(
                " (observed at simulation second ",
                maxsplit=1,
            )[0]

            if observed_value not in user_message:
                raise TemporalDatasetValidationError(
                    f"{example.example_id}: user message omits a source observation."
                )

            if observed_value not in assistant_message:
                raise TemporalDatasetValidationError(
                    f"{example.example_id}: assistant narration omits "
                    "a source observation."
                )

        if str(example.initial_observed_at_seconds) not in user_message:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: initial timestamp missing from user message."
            )

        if str(example.follow_up_observed_at_seconds) not in user_message:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: follow-up timestamp missing from user message."
            )

    @classmethod
    def _validate_no_hidden_state(
        cls,
        example: TemporalSceneNarrationExample,
    ) -> None:
        serialized = " ".join(message.content for message in example.messages)

        for pattern in cls.FORBIDDEN_PATTERNS:
            if pattern.search(serialized):
                raise TemporalDatasetValidationError(
                    f"{example.example_id}: forbidden internal "
                    f"simulator information found: {pattern.pattern}"
                )
