import pytest

from sceneops.training.dataset import ChatMessage
from sceneops.training.temporal_scene_narration_dataset import (
    TemporalSceneNarrationDatasetGenerator,
)
from sceneops.training.temporal_scene_narration_validator import (
    TemporalDatasetValidationError,
    TemporalSceneNarrationValidator,
)


@pytest.fixture
def examples():
    return TemporalSceneNarrationDatasetGenerator().generate()


def test_valid_temporal_dataset_passes(examples):
    result = TemporalSceneNarrationValidator.validate(examples)

    assert result == {
        "examples": 10,
        "unique_scenarios": 5,
        "unique_patients": 2,
        "unique_example_ids": 10,
    }


def test_empty_dataset_is_rejected():
    with pytest.raises(
        TemporalDatasetValidationError,
        match="cannot be empty",
    ):
        TemporalSceneNarrationValidator.validate([])


def test_duplicate_example_ids_are_rejected(examples):
    duplicate = examples[0].model_copy(deep=True)

    invalid = [examples[0], duplicate]

    with pytest.raises(
        TemporalDatasetValidationError,
        match="Duplicate",
    ):
        TemporalSceneNarrationValidator.validate(invalid)


def test_invalid_follow_up_timestamp_is_rejected(examples):
    invalid_example = examples[0].model_copy(
        update={
            "follow_up_observed_at_seconds": (examples[0].initial_observed_at_seconds)
        }
    )

    with pytest.raises(
        TemporalDatasetValidationError,
        match="follow-up timestamp",
    ):
        TemporalSceneNarrationValidator.validate([invalid_example])


def test_source_fact_timestamp_mismatch_is_rejected(examples):
    example = examples[0]
    invalid_fact = example.initial_facts[0].model_copy(
        update={"value": "35 (observed at simulation second 999)"}
    )
    invalid_example = example.model_copy(update={"initial_facts": (invalid_fact,)})

    with pytest.raises(
        TemporalDatasetValidationError,
        match="timestamp",
    ):
        TemporalSceneNarrationValidator.validate([invalid_example])


def test_missing_observation_in_assistant_message_is_rejected(examples):
    example = examples[0]

    invalid_assistant = ChatMessage(
        role="assistant",
        content="The patient was observed at two different times.",
    )

    invalid_example = example.model_copy(
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
        match="assistant narration omits",
    ):
        TemporalSceneNarrationValidator.validate([invalid_example])


def test_hidden_event_id_in_message_is_rejected(examples):
    example = examples[0]

    invalid_assistant = ChatMessage(
        role="assistant",
        content=example.messages[2].content + " Internal event EV001.",
    )

    invalid_example = example.model_copy(
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
        TemporalSceneNarrationValidator.validate([invalid_example])


def test_invalid_source_fact_identity_is_rejected(examples):
    example = examples[0]
    invalid_fact = example.initial_facts[0].model_copy(
        update={"fact": "internal_patient_state:P005:respiratory_rate"}
    )

    invalid_example = example.model_copy(update={"initial_facts": (invalid_fact,)})

    with pytest.raises(
        TemporalDatasetValidationError,
        match="source fact identity",
    ):
        TemporalSceneNarrationValidator.validate([invalid_example])


def test_generation_and_validation_are_reproducible(examples):
    first = TemporalSceneNarrationDatasetGenerator().generate()
    second = TemporalSceneNarrationDatasetGenerator().generate()

    assert [example.model_dump() for example in first] == [
        example.model_dump() for example in second
    ]

    assert TemporalSceneNarrationValidator.validate(
        first
    ) == TemporalSceneNarrationValidator.validate(second)
