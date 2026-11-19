import pytest

from sceneops.simulator.observations import ObservationType
from sceneops.training.temporal_scene_narration_dataset_v2 import (
    TEMPORAL_SCENE_GENERATOR_V2_VERSION,
    TemporalSceneNarrationDatasetV2Generator,
)


@pytest.fixture
def generator():
    return TemporalSceneNarrationDatasetV2Generator()


def test_generates_twenty_examples(generator):
    examples = generator.generate()
    assert len(examples) == 20


def test_covers_five_scenarios(generator):
    examples = generator.generate()
    assert len({item.scenario_id for item in examples}) == 5


def test_each_scenario_has_four_temporal_targets(generator):
    examples = generator.generate()

    for scenario_id in {item.scenario_id for item in examples}:
        scenario_examples = [
            item for item in examples if item.scenario_id == scenario_id
        ]

        assert len(scenario_examples) == 4
        assert {item.patient_id for item in scenario_examples} == {
            "P005",
            "P006",
            "P007",
            "P008",
        }


def test_covers_four_observation_types(generator):
    examples = generator.generate()

    assert {item.observation_type for item in examples} == {
        ObservationType.RESPIRATORY_RATE,
        ObservationType.FOLLOWS_COMMANDS,
        ObservationType.RADIAL_PULSE,
        ObservationType.MENTAL_STATUS,
    }


def test_follow_up_observations_are_later(generator):
    examples = generator.generate()

    assert all(
        item.follow_up_observed_at_seconds > item.initial_observed_at_seconds
        for item in examples
    )


def test_expected_observation_changes(generator):
    examples = generator.generate()

    expected = {
        "P005": ("20", "35"),
        "P006": ("follows commands", "does not follow commands"),
        "P007": ("present", "absent"),
        "P008": ("alert", "confused"),
    }

    for example in examples:
        initial = example.initial_facts[0].value.split(
            " (observed at simulation second ", maxsplit=1
        )[0]
        follow_up = example.follow_up_facts[0].value.split(
            " (observed at simulation second ", maxsplit=1
        )[0]

        if example.patient_id == "P005":
            assert int(initial) in (20, 21, 22)
            assert follow_up == expected["P005"][1]
        else:
            assert initial == expected[example.patient_id][0]
            assert follow_up == expected[example.patient_id][1]


def test_generator_version_is_v2(generator):
    examples = generator.generate()

    assert all(
        item.generator_version == TEMPORAL_SCENE_GENERATOR_V2_VERSION
        for item in examples
    )


def test_training_records_exclude_internal_metadata(generator):
    examples = generator.generate()

    for example in examples:
        record = example.to_training_record()

        assert set(record) == {"messages"}

        serialized = " ".join(message["content"] for message in record["messages"])

        assert "TEMP-EV003" not in serialized
        assert "TEMP-EV004" not in serialized
        assert "patient_state_changed" not in serialized
        assert "scheduled deterioration event" not in serialized.lower()


def test_generation_is_reproducible(generator):
    first = generator.generate()
    second = generator.generate()

    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]


def test_duplicate_scenario_indices_are_rejected(generator):
    with pytest.raises(ValueError, match="unique"):
        generator.generate((1, 1))


def test_empty_scenario_indices_are_rejected(generator):
    with pytest.raises(ValueError, match="At least one"):
        generator.generate(())
