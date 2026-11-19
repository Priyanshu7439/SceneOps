import hashlib
import json

from scripts.generate_temporal_dataset import generate_and_export


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def test_export_creates_all_three_files(tmp_path):
    manifest = generate_and_export(tmp_path)

    assert (tmp_path / manifest["master_file"]).exists()
    assert (tmp_path / manifest["training_file"]).exists()
    assert (tmp_path / "temporal_scene_narration_v1_manifest.json").exists()


def test_export_contains_ten_examples(tmp_path):
    manifest = generate_and_export(tmp_path)

    master_records = read_jsonl(tmp_path / manifest["master_file"])
    training_records = read_jsonl(tmp_path / manifest["training_file"])

    assert len(master_records) == 10
    assert len(training_records) == 10
    assert manifest["total_examples"] == 10


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

    assert manifest["total_examples"] == 2
    assert manifest["unique_scenarios"] == 1
