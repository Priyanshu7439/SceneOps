import pytest

from sceneops.training.dataset import PatientDialogueDatasetGenerator
from sceneops.training.splitting import split_by_scenario


@pytest.fixture
def examples():
    return PatientDialogueDatasetGenerator().generate()


def test_splits_preserve_all_examples(examples):
    splits = split_by_scenario(examples)

    total = sum(len(split) for split in splits.values())

    assert total == len(examples)
    assert total == 5520


def test_scenarios_do_not_leak_between_splits(examples):
    splits = split_by_scenario(examples)

    scenario_sets = {
        name: {example.scenario_id for example in split}
        for name, split in splits.items()
    }

    assert scenario_sets["train"].isdisjoint(scenario_sets["validation"])
    assert scenario_sets["train"].isdisjoint(scenario_sets["test"])
    assert scenario_sets["validation"].isdisjoint(scenario_sets["test"])


def test_each_scenario_belongs_to_exactly_one_split(examples):
    splits = split_by_scenario(examples)

    assignments = {}

    for split_name, split_examples in splits.items():
        for example in split_examples:
            assignments.setdefault(
                example.scenario_id,
                set(),
            ).add(split_name)

    assert len(assignments) == 92
    assert all(len(split_names) == 1 for split_names in assignments.values())


def test_split_is_reproducible(examples):
    first = split_by_scenario(examples, seed=42)
    second = split_by_scenario(examples, seed=42)

    for name in ("train", "validation", "test"):
        assert [example.example_id for example in first[name]] == [
            example.example_id for example in second[name]
        ]


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        split_by_scenario([])


def test_less_than_three_scenarios_is_rejected(examples):
    one_scenario = [example for example in examples if example.scenario_id == "MCI-001"]

    with pytest.raises(ValueError, match="At least 3"):
        split_by_scenario(one_scenario)
