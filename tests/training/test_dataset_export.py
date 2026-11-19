import importlib.util
from pathlib import Path

import pytest

from sceneops.training.dataset import (
    ChatMessage,
    DatasetExample,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_dataset.py"

SPEC = importlib.util.spec_from_file_location(
    "generate_dataset",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError("Could not load generate_dataset.py")

GENERATE_DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATE_DATASET)

calculate_sha256 = GENERATE_DATASET.calculate_sha256
load_jsonl = GENERATE_DATASET.load_jsonl
publish_outputs = GENERATE_DATASET.publish_outputs
write_jsonl = GENERATE_DATASET.write_jsonl


def create_example() -> DatasetExample:
    return DatasetExample(
        example_id="MCI-001-P001-Q001",
        scenario_id="MCI-001",
        patient_id="P001",
        task="patient_dialogue",
        generator_version="2.0.0",
        disclosure_fact="reported_pain_level",
        disclosure_value="5",
        messages=(
            ChatMessage(
                role="system",
                content="You are a synthetic patient.",
            ),
            ChatMessage(
                role="user",
                content="How much pain are you experiencing?",
            ),
            ChatMessage(
                role="assistant",
                content="My pain is 5.",
            ),
        ),
    )


def test_calculate_sha256_returns_correct_checksum(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("SceneOps checksum test", encoding="utf-8")

    expected = "f6c7f6e0e3f3f8d6a7f2b6b8a0d1c5e6"

    import hashlib

    expected = hashlib.sha256(b"SceneOps checksum test").hexdigest()

    assert calculate_sha256(file_path) == expected


def test_jsonl_write_and_load_round_trip(tmp_path: Path):
    file_path = tmp_path / "dataset.jsonl"
    original = [create_example()]

    write_jsonl(file_path, original)
    loaded = load_jsonl(file_path)

    assert loaded == original


def test_load_jsonl_rejects_empty_records(tmp_path: Path):
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Empty JSONL record"):
        load_jsonl(file_path)


def test_publish_outputs_replaces_staged_files(tmp_path: Path):
    staged = tmp_path / "staged"
    output = tmp_path / "output"

    staged.mkdir()
    output.mkdir()

    staged_master = staged / "master.jsonl"
    staged_train = staged / "train.jsonl"

    output_master = output / "master.jsonl"
    output_train = output / "train.jsonl"

    staged_master.write_text("new master", encoding="utf-8")
    staged_train.write_text("new train", encoding="utf-8")

    output_master.write_text("old master", encoding="utf-8")
    output_train.write_text("old train", encoding="utf-8")

    publish_outputs(
        {
            "master": staged_master,
            "train": staged_train,
        },
        {
            "master": output_master,
            "train": output_train,
        },
    )

    assert output_master.read_text(encoding="utf-8") == "new master"
    assert output_train.read_text(encoding="utf-8") == "new train"

    assert not staged_master.exists()
    assert not staged_train.exists()


def test_publish_outputs_rolls_back_on_replacement_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    staged = tmp_path / "staged"
    output = tmp_path / "output"

    staged.mkdir()
    output.mkdir()

    staged_master = staged / "master.jsonl"
    staged_train = staged / "train.jsonl"

    output_master = output / "master.jsonl"
    output_train = output / "train.jsonl"

    staged_master.write_text("new master", encoding="utf-8")
    staged_train.write_text("new train", encoding="utf-8")

    output_master.write_text("old master", encoding="utf-8")
    output_train.write_text("old train", encoding="utf-8")

    original_replace = Path.replace

    def failing_replace(
        self: Path,
        target: str | Path,
    ) -> Path:
        if self == staged_train:
            raise OSError("Simulated publishing failure.")

        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="Simulated publishing failure"):
        publish_outputs(
            {
                "master": staged_master,
                "train": staged_train,
            },
            {
                "master": output_master,
                "train": output_train,
            },
        )

    assert output_master.read_text(encoding="utf-8") == "old master"
    assert output_train.read_text(encoding="utf-8") == "old train"
    assert not Path(str(output_master) + ".bak.tmp").exists()
