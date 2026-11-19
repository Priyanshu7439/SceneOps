from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from sceneops.simulator.observations import ObservationType
from sceneops.training.temporal_scene_narration_dataset import (
    TemporalSceneNarrationExample,
)
from sceneops.training.temporal_scene_narration_dataset_v2 import (
    TEMPORAL_SCENE_GENERATOR_V2_VERSION,
    TEMPORAL_V2_TARGETS,
)
from sceneops.training.temporal_scene_narration_validator import (
    TemporalDatasetValidationError,
    TemporalSceneNarrationValidator,
)


class TemporalSceneNarrationV2Validator:
    """Validate expanded temporal narration examples and provenance."""

    EXPECTED_INITIAL_VALUES: ClassVar[dict[tuple[str, ObservationType], set[str]]] = {
        ("P005", ObservationType.RESPIRATORY_RATE): {"20", "21", "22"},
        ("P006", ObservationType.FOLLOWS_COMMANDS): {"follows commands"},
        ("P007", ObservationType.RADIAL_PULSE): {"present"},
        ("P008", ObservationType.MENTAL_STATUS): {"alert"},
    }

    EXPECTED_FOLLOW_UP_VALUES: ClassVar[dict[tuple[str, ObservationType], set[str]]] = {
        ("P005", ObservationType.RESPIRATORY_RATE): {"35"},
        ("P006", ObservationType.FOLLOWS_COMMANDS): {"does not follow commands"},
        ("P007", ObservationType.RADIAL_PULSE): {"absent"},
        ("P008", ObservationType.MENTAL_STATUS): {"confused"},
    }

    @classmethod
    def validate(
        cls,
        examples: list[TemporalSceneNarrationExample],
    ) -> dict[str, int]:
        if not examples:
            raise TemporalDatasetValidationError("Temporal V2 dataset cannot be empty.")

        # Check duplicate example IDs before all other validation.
        example_ids = [example.example_id for example in examples]

        if len(example_ids) != len(set(example_ids)):
            raise TemporalDatasetValidationError(
                "Duplicate temporal example IDs found."
            )

        # Validate every individual example against V2 and V1 rules.
        # This catches hidden information even in incomplete test subsets.
        for example in examples:
            cls._validate_v2_example(example)
            TemporalSceneNarrationValidator.validate([example])

        grouped: dict[
            str,
            list[TemporalSceneNarrationExample],
        ] = defaultdict(list)

        for example in examples:
            grouped[example.scenario_id].append(example)

        expected_targets = set(TEMPORAL_V2_TARGETS)

        for scenario_id, scenario_examples in grouped.items():
            actual_targets = [
                (example.patient_id, example.observation_type)
                for example in scenario_examples
            ]

            if len(actual_targets) != len(set(actual_targets)):
                raise TemporalDatasetValidationError(
                    f"{scenario_id}: duplicate temporal target found."
                )

            if set(actual_targets) != expected_targets:
                raise TemporalDatasetValidationError(
                    f"{scenario_id}: expected exactly four approved temporal targets."
                )

        return {
            "examples": len(examples),
            "unique_scenarios": len(grouped),
            "unique_patients": len({example.patient_id for example in examples}),
            "unique_example_ids": len(set(example_ids)),
            "unique_temporal_targets": len(
                {(example.patient_id, example.observation_type) for example in examples}
            ),
            "generator_version": 2,
        }

    @classmethod
    def _validate_v2_example(
        cls,
        example: TemporalSceneNarrationExample,
    ) -> None:
        if example.generator_version != TEMPORAL_SCENE_GENERATOR_V2_VERSION:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: incorrect generator version."
            )

        target = (example.patient_id, example.observation_type)

        if target not in set(TEMPORAL_V2_TARGETS):
            raise TemporalDatasetValidationError(
                f"{example.example_id}: unapproved patient/observation pair."
            )

        if len(example.initial_facts) != 1:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: expected exactly one initial fact."
            )

        if len(example.follow_up_facts) != 1:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: expected exactly one follow-up fact."
            )

        initial_value = cls._extract_observed_value(example.initial_facts[0].value)
        follow_up_value = cls._extract_observed_value(example.follow_up_facts[0].value)

        if initial_value not in cls.EXPECTED_INITIAL_VALUES[target]:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: unexpected initial observation."
            )

        if follow_up_value not in cls.EXPECTED_FOLLOW_UP_VALUES[target]:
            raise TemporalDatasetValidationError(
                f"{example.example_id}: unexpected follow-up observation."
            )

    @staticmethod
    def _extract_observed_value(value: str) -> str:
        marker = " (observed at simulation second "

        if marker not in value or not value.endswith(")"):
            raise TemporalDatasetValidationError(
                "Source fact has an invalid observation timestamp format."
            )

        return value.split(marker, maxsplit=1)[0]
