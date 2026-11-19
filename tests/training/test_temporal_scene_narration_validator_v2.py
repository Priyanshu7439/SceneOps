import pytest

from sceneops.simulator.observations import ObservationType
from sceneops.training.temporal_scene_narration_dataset_v2 import (
    TemporalSceneNarrationDatasetV2Generator,
)
from sceneops.training.temporal_scene_narration_validator import (
    TemporalDatasetValidationError,
)
from sceneops.training.temporal_scene_narration_validator_v2 import (
    TemporalSceneNarrationV2Validator,
)


@pytest.fixture
def examples():
    return TemporalSceneNarrationDatasetV2Generator().generate()


def test_valid_v2_dataset_passes(examples):
    result = TemporalSceneNarrationV2Validator.validate(examples)

    assert result["examples"] == 20
    assert result["unique_scenarios"] == 5
    assert result["unique_temporal_targets"] == 4
    assert result["generator_version"] == 2


def test_empty_dataset_is_rejected():
    with pytest.raises(
        TemporalDatasetValidationError,
        match="cannot be empty",
    ):
        TemporalSceneNarrationV2Validator.validate([])


def test_duplicate_example_ids_are_rejected(examples):
    duplicate = examples[0].model_copy(deep=True)

    with pytest.raises(
        TemporalDatasetValidationError,
        match="Duplicate",
    ):
        TemporalSceneNarrationV2Validator.validate([examples[0], duplicate])


def test_wrong_generator_version_is_rejected(examples):
    invalid = examples[0].model_copy(update={"generator_version": "1.0.0"})

    with pytest.raises(
        TemporalDatasetValidationError,
        match="generator version",
    ):
        TemporalSceneNarrationV2Validator.validate([invalid])


def test_unapproved_patient_observation_pair_is_rejected(examples):
    invalid = examples[0].model_copy(
        update={
            "patient_id": "P020",
            "observation_type": ObservationType.RADIAL_PULSE,
        }
    )

    with pytest.raises(
        TemporalDatasetValidationError,
        match="unapproved",
    ):
        TemporalSceneNarrationV2Validator.validate([invalid])


def test_unexpected_initial_value_is_rejected(examples):
    example = examples[0]
    invalid_fact = example.initial_facts[0].model_copy(
        update={"value": "99 (observed at simulation second 0)"}
    )
    invalid = example.model_copy(update={"initial_facts": (invalid_fact,)})

    with pytest.raises(
        TemporalDatasetValidationError,
        match="unexpected initial",
    ):
        TemporalSceneNarrationV2Validator.validate([invalid])


def test_unexpected_follow_up_value_is_rejected(examples):
    example = examples[0]
    invalid_fact = example.follow_up_facts[0].model_copy(
        update={"value": "999 (observed at simulation second 120)"}
    )
    invalid = example.model_copy(update={"follow_up_facts": (invalid_fact,)})

    with pytest.raises(
        TemporalDatasetValidationError,
        match="unexpected follow-up",
    ):
        TemporalSceneNarrationV2Validator.validate([invalid])


def test_scenario_missing_temporal_target_is_rejected(examples):
    invalid = [
        example
        for example in examples
        if not (example.scenario_id == "MCI-001" and example.patient_id == "P008")
    ]

    with pytest.raises(
        TemporalDatasetValidationError,
        match="exactly four",
    ):
        TemporalSceneNarrationV2Validator.validate(invalid)


def test_duplicate_temporal_target_is_rejected(examples):
    duplicate = examples[0].model_copy(update={"example_id": "MCI-001-extra"})

    invalid = [
        *examples,
        duplicate,
    ]

    with pytest.raises(
        TemporalDatasetValidationError,
        match="duplicate temporal target",
    ):
        TemporalSceneNarrationV2Validator.validate(invalid)


def test_hidden_event_id_is_rejected(examples):
    example = examples[0]
    invalid_assistant = example.messages[2].model_copy(
        update={"content": example.messages[2].content + " Internal event EV999."}
    )

    invalid = example.model_copy(
        update={
            "messages": (
                example.messages[0],
                example.messages[1],
                invalid_assistant,
            )
        }
    )

    with pytest.raises(
        TemporalDatasetValidationError,
        match="forbidden internal",
    ):
        TemporalSceneNarrationV2Validator.validate([invalid])


def test_generation_and_validation_are_reproducible(examples):
    second = TemporalSceneNarrationDatasetV2Generator().generate()

    first_result = TemporalSceneNarrationV2Validator.validate(examples)
    second_result = TemporalSceneNarrationV2Validator.validate(second)

    assert first_result == second_result
