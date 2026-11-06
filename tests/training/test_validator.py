import pytest

from sceneops.training.dataset import (
    ChatMessage,
    PatientDialogueDatasetGenerator,
)
from sceneops.training.validator import (
    DatasetValidationError,
    PatientDialogueDatasetValidator,
)


@pytest.fixture
def examples():
    return PatientDialogueDatasetGenerator().generate()


@pytest.fixture
def validator():
    return PatientDialogueDatasetValidator()


def test_valid_dataset_passes(examples, validator):
    report = validator.validate(examples)

    assert report["examples"] == 60
    assert report["unique_patients"] == 20
    assert report["unique_scenarios"] == 1
    assert report["unique_example_ids"] == 60


def test_empty_dataset_is_rejected(validator):
    with pytest.raises(DatasetValidationError):
        validator.validate([])


def test_duplicate_example_ids_are_rejected(
    examples,
    validator,
):
    corrupted = list(examples)
    corrupted[1] = corrupted[1].model_copy(
        update={"example_id": corrupted[0].example_id}
    )

    with pytest.raises(
        DatasetValidationError,
        match="Duplicate example IDs",
    ):
        validator.validate(corrupted)


def test_unauthorized_disclosure_is_rejected(
    examples,
    validator,
):
    corrupted = list(examples)
    corrupted[0] = corrupted[0].model_copy(
        update={"disclosure_fact": "hidden_diagnosis"}
    )

    with pytest.raises(
        DatasetValidationError,
        match="unauthorized disclosure fact",
    ):
        validator.validate(corrupted)


def test_mismatched_disclosure_value_is_rejected(
    examples,
    validator,
):
    corrupted = list(examples)
    corrupted[0] = corrupted[0].model_copy(
        update={"disclosure_value": "10/10"}
    )

    with pytest.raises(
        DatasetValidationError,
        match="does not match metadata",
    ):
        validator.validate(corrupted)


def test_incorrect_assistant_answer_is_rejected(
    examples,
    validator,
):
    corrupted = list(examples)
    original = corrupted[0]

    messages = list(original.messages)
    messages[2] = ChatMessage(
        role="assistant",
        content="I have a severe head injury.",
    )

    corrupted[0] = original.model_copy(
        update={"messages": tuple(messages)}
    )

    with pytest.raises(
        DatasetValidationError,
        match="answer does not match",
    ):
        validator.validate(corrupted)