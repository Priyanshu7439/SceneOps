import hashlib
import json

from scripts.generate_temporal_dataset_v2 import generate_and_export


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def test_export_creates_all_three_files(tmp_path):
    manifest = generate_and_export(tmp_path)

    assert (tmp_path / manifest["master_file"]).exists()
    assert (tmp_path / manifest["training_file"]).exists()
    assert (tmp_path / "temporal_scene_narration_v2_manifest.json").exists()


def test_export_contains_twenty_examples(tmp_path):
    manifest = generate_and_export(tmp_path)

    master_records = read_jsonl(tmp_path / manifest["master_file"])
    training_records = read_jsonl(tmp_path / manifest["training_file"])

    assert len(master_records) == 20
    assert len(training_records) == 20
    assert manifest["total_examples"] == 20


def test_training_export_contains_only_messages(tmp_path):
    manifest = generate_and_export(tmp_path)

    records = read_jsonl(tmp_path / manifest["training_file"])

    for record in records:
        assert set(record) == {"messages"}
        assert len(record["messages"]) == 3


def test_manifest_contains_valid_checksums(tmp_path):
    manifest = generate_and_export(tmp_path)

    master_path = tmp_path / manifest["master_file"]
    training_path = tmp_path / manifest["training_file"]

    master_hash = hashlib.sha256(master_path.read_bytes()).hexdigest()

    training_hash = hashlib.sha256(training_path.read_bytes()).hexdigest()

    assert manifest["master_sha256"] == master_hash
    assert manifest["training_sha256"] == training_hash
    assert manifest["validation_status"] == "passed"
    assert manifest["split_status"] == "not_split"


def test_export_supports_a_single_scenario(tmp_path):
    manifest = generate_and_export(
        tmp_path,
        scenario_indices=(1,),
    )

    assert manifest["total_examples"] == 4
    assert manifest["unique_scenarios"] == 1
    assert manifest["unique_temporal_targets"] == 4


def test_each_scenario_contains_four_temporal_targets(tmp_path):
    manifest = generate_and_export(tmp_path)

    records = read_jsonl(tmp_path / manifest["master_file"])

    targets_by_scenario = {}

    for record in records:
        scenario_id = record["scenario_id"]
        target = (
            record["patient_id"],
            record["observation_type"],
        )

        targets_by_scenario.setdefault(scenario_id, set()).add(target)

    assert len(targets_by_scenario) == 5

    for targets in targets_by_scenario.values():
        assert len(targets) == 4


def test_export_does_not_leave_temporary_files(tmp_path):
    generate_and_export(tmp_path)

    temporary_files = list(tmp_path.glob("*.tmp"))

    assert temporary_files == []
