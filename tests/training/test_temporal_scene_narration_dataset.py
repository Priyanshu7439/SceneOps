import pytest

from sceneops.simulator.observations import ObservationType
from sceneops.training.temporal_scene_narration_dataset import (
    TemporalSceneNarrationDatasetGenerator,
)


@pytest.fixture
def generator():
    return TemporalSceneNarrationDatasetGenerator()


def test_generates_ten_temporal_examples(generator):
    examples = generator.generate()

    assert len(examples) == 10


def test_examples_cover_five_scenarios(generator):
    examples = generator.generate()

    assert len({example.scenario_id for example in examples}) == 5


def test_each_scenario_has_two_temporal_targets(generator):
    examples = generator.generate()

    for scenario_id in {example.scenario_id for example in examples}:
        scenario_examples = [
            example for example in examples if example.scenario_id == scenario_id
        ]

        assert len(scenario_examples) == 2
        assert {example.patient_id for example in scenario_examples} == {"P005", "P006"}


def test_initial_observations_are_at_second_zero(generator):
    examples = generator.generate()

    assert all(example.initial_observed_at_seconds == 0 for example in examples)


def test_follow_up_observations_are_later(generator):
    examples = generator.generate()

    assert all(
        example.follow_up_observed_at_seconds > example.initial_observed_at_seconds
        for example in examples
    )


def test_p005_respiratory_rate_changes(generator):
    examples = generator.generate()

    p005_examples = [example for example in examples if example.patient_id == "P005"]

    assert len(p005_examples) == 5

    for example in p005_examples:
        assert example.observation_type == ObservationType.RESPIRATORY_RATE

        initial_value = example.initial_facts[0].value
        follow_up_value = example.follow_up_facts[0].value

        assert "observed at simulation second 0" in initial_value
        assert "35" in follow_up_value


def test_p006_command_following_changes(generator):
    examples = generator.generate()

    p006_examples = [example for example in examples if example.patient_id == "P006"]

    assert len(p006_examples) == 5

    for example in p006_examples:
        assert example.observation_type == ObservationType.FOLLOWS_COMMANDS
        assert "follows commands" in example.initial_facts[0].value
        assert "does not follow commands" in example.follow_up_facts[0].value


def test_training_record_excludes_internal_metadata(generator):
    examples = generator.generate()

    for example in examples:
        record = example.to_training_record()

        assert set(record) == {"messages"}

        serialized_messages = " ".join(
            message["content"] for message in record["messages"]
        )

        assert "EV001" not in serialized_messages
        assert "EV002" not in serialized_messages
        assert "patient_state_changed" not in serialized_messages
        assert "deterioration event" not in serialized_messages


def test_training_messages_use_only_observed_values(generator):
    examples = generator.generate()

    for example in examples:
        user_message = example.messages[1].content
        assistant_message = example.messages[2].content

        initial_value = example.initial_facts[0].value.split(
            " (observed at simulation second "
        )[0]

        follow_up_value = example.follow_up_facts[0].value.split(
            " (observed at simulation second "
        )[0]

        assert initial_value in user_message
        assert follow_up_value in user_message
        assert initial_value in assistant_message
        assert follow_up_value in assistant_message


def test_generation_is_reproducible(generator):
    first = generator.generate()
    second = generator.generate()

    assert [example.model_dump() for example in first] == [
        example.model_dump() for example in second
    ]


def test_duplicate_scenario_indices_are_rejected(generator):
    with pytest.raises(ValueError, match="unique"):
        generator.generate((1, 1))


def test_empty_scenario_indices_are_rejected(generator):
    with pytest.raises(ValueError, match="At least one"):
        generator.generate(())
