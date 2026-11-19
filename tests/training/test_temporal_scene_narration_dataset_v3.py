import pytest

from sceneops.simulator.observations import ObservationType
from sceneops.training.temporal_scene_narration_dataset_v3 import (
    FIELD_OBSERVATION_MAP,
    TEMPORAL_SCENE_GENERATOR_V3_VERSION,
    TemporalSceneNarrationDatasetV3Generator,
)


@pytest.fixture
def generator():
    return TemporalSceneNarrationDatasetV3Generator()


def test_generates_six_examples_for_one_seed(generator):
    examples = generator.generate((42,))

    assert len(examples) == 6


def test_covers_five_observation_types(generator):
    examples = generator.generate((42,))

    assert {item.observation_type for item in examples} == {
        ObservationType.AMBULATORY_STATUS,
        ObservationType.RADIAL_PULSE,
        ObservationType.RESPIRATORY_RATE,
        ObservationType.MENTAL_STATUS,
        ObservationType.FOLLOWS_COMMANDS,
    }


def test_every_example_has_valid_chronology(generator):
    examples = generator.generate((42,))

    for example in examples:
        assert (
            example.follow_up_observed_at_seconds > example.initial_observed_at_seconds
        )


def test_event_examples_have_expected_follow_up_timestamps(generator):
    examples = generator.generate((42,))

    expected_windows = {
        "ambulatory_status": ((60, 120),),
        "radial_pulse": ((150, 210),),
        "respiratory_rate": ((240, 300),),
        "mental_status": ((330, 390), (420, 480)),
        "follows_commands": ((420, 480),),
    }

    for example in examples:
        windows = expected_windows[example.observation_type.value]
        timestamp = example.follow_up_observed_at_seconds

        assert any(minimum <= timestamp <= maximum for minimum, maximum in windows)

    mental_status_examples = [
        item for item in examples if item.observation_type.value == "mental_status"
    ]

    assert len(mental_status_examples) == 2


def test_multiple_changes_create_separate_examples(generator):
    examples = generator.generate((42,))

    patient_counts = {}

    for example in examples:
        patient_counts[example.patient_id] = (
            patient_counts.get(example.patient_id, 0) + 1
        )

    assert sorted(patient_counts.values()) == [1, 1, 1, 1, 2]


def test_example_ids_are_unique(generator):
    examples = generator.generate((42,))

    example_ids = [item.example_id for item in examples]

    assert len(example_ids) == len(set(example_ids))


def test_generator_version_is_v3(generator):
    examples = generator.generate((42,))

    assert all(
        item.generator_version == TEMPORAL_SCENE_GENERATOR_V3_VERSION
        for item in examples
    )


def test_training_records_exclude_internal_metadata(generator):
    examples = generator.generate((42,))

    for example in examples:
        record = example.to_training_record()

        assert set(record) == {"messages"}

        serialized = " ".join(message["content"] for message in record["messages"])

        assert "TEMP-EV" not in serialized
        assert "patient_state_changed" not in serialized
        assert "scheduled deterioration event" not in serialized.lower()


def test_generation_is_reproducible(generator):
    first = generator.generate((42, 43))
    second = generator.generate((42, 43))

    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]


def test_multiple_seeds_generate_expected_count(generator):
    examples = generator.generate((42, 43, 44))

    assert len(examples) == 18
    assert len({item.scenario_id for item in examples}) == 3


def test_duplicate_seeds_are_rejected(generator):
    with pytest.raises(ValueError, match="unique"):
        generator.generate((42, 42))


def test_empty_seeds_are_rejected(generator):
    with pytest.raises(ValueError, match="At least one"):
        generator.generate(())


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42"])
def test_invalid_seeds_are_rejected(generator, seed):
    with pytest.raises(ValueError):
        generator.generate((seed,))


def test_observation_mapping_covers_all_supported_changes():
    assert set(FIELD_OBSERVATION_MAP) == {
        "respiratory_rate",
        "ambulatory",
        "radial_pulse_present",
        "follows_commands",
        "major_hemorrhage",
        "mental_status",
    }


def test_full_dataset_generates_two_objectives_per_patient(generator):
    examples = generator.generate_full_dataset((42,))

    assert len(examples) == 40
    assert sum(item.task == "scene_narration" for item in examples) == 20
    assert sum(item.task == "temporal_scene_narration" for item in examples) == 20


def test_full_panel_temporal_examples_include_all_approved_observations(generator):
    examples = generator.generate_full_dataset((42,))
    temporal = [item for item in examples if item.task == "temporal_scene_narration"]

    assert len(temporal) == 20
    assert all(len(item.observation_types) >= 5 for item in temporal)
    assert all(
        item.initial_observed_at_seconds == 0
        and item.follow_up_observed_at_seconds == 600
        for item in temporal
    )
    assert all(
        {fact.fact.rsplit(":", maxsplit=1)[-1] for fact in item.initial_facts}
        == {fact.fact.rsplit(":", maxsplit=1)[-1] for fact in item.follow_up_facts}
        for item in temporal
    )


def test_full_dataset_generation_is_reproducible(generator):
    first = generator.generate_full_dataset((42, 43))
    second = generator.generate_full_dataset((42, 43))

    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]


def test_diverse_scenario_events_vary_by_seed():
    from sceneops.training.temporal_scenario_generator import (
        TemporalScenarioGenerator,
    )

    scenario_generator = TemporalScenarioGenerator()
    scenarios = [scenario_generator.generate_diverse(seed) for seed in range(12)]
    respiratory_targets = {
        event.changes.respiratory_rate
        for scenario in scenarios
        for event in scenario.deterioration_events
        if event.changes.respiratory_rate is not None
    }

    assert len(respiratory_targets) > 4


def test_exact_deduplication_removes_repeated_training_records(generator):
    examples = generator.generate_full_dataset((42,))
    duplicated = examples + examples[:3]

    unique = generator.deduplicate_exact_examples(duplicated)

    assert len(unique) == len(examples)
    assert [item.example_id for item in unique] == [
        item.example_id for item in examples
    ]


def test_full_dataset_validator_reports_both_objectives(generator):
    examples = generator.generate_full_dataset((42, 43))

    report = generator.validate_full_dataset(
        examples,
        expected_scenarios=2,
    )

    assert report["total_examples"] == 80
    assert report["scene_narration_examples"] == 40
    assert report["temporal_narration_examples"] == 40
    assert report["unique_example_ids"] == 80


def test_exporter_writes_master_splits_and_manifest(tmp_path):
    import hashlib
    import json

    from sceneops.training.temporal_scene_narration_dataset_v3 import (
        generate_and_export,
    )

    manifest = generate_and_export(
        tmp_path,
        seeds=(42, 43, 44),
        minimum_examples=100,
    )

    expected_files = {
        "sceneops_narration_master.jsonl",
        "sceneops_narration_train.jsonl",
        "sceneops_narration_validation.jsonl",
        "sceneops_narration_test.jsonl",
        "sceneops_narration_manifest.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}
    assert manifest["total_examples"] >= 100
    assert manifest["validation_status"] == "passed"
    assert manifest["split_status"] == "passed"

    for key in ("master", "train", "validation", "test"):
        filename = manifest["files"][key]
        digest = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        assert manifest["sha256"][key] == digest

    manifest_record = json.loads(
        (tmp_path / "sceneops_narration_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_record["total_examples"] == manifest["total_examples"]


def test_exporter_rejects_dataset_below_required_minimum(tmp_path):
    from sceneops.training.temporal_scene_narration_dataset_v3 import (
        generate_and_export,
    )

    with pytest.raises(ValueError, match="minimum required"):
        generate_and_export(
            tmp_path,
            seeds=(42, 43, 44),
            minimum_examples=1000,
        )
