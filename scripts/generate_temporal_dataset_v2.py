from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sceneops.training.temporal_scene_narration_dataset_v2 import (
    TEMPORAL_SCENE_GENERATOR_V2_VERSION,
    TemporalSceneNarrationDatasetV2Generator,
)
from sceneops.training.temporal_scene_narration_validator_v2 import (
    TemporalSceneNarrationV2Validator,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSON records as UTF-8 JSONL."""

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def generate_and_export(
    output_dir: Path,
    scenario_indices: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    """Generate, validate and export temporal narration V2."""

    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TemporalSceneNarrationDatasetV2Generator()
    examples = generator.generate(scenario_indices)

    validation = TemporalSceneNarrationV2Validator.validate(examples)

    master_path = output_dir / "temporal_scene_narration_v2.jsonl"
    training_path = output_dir / "temporal_scene_narration_v2_training.jsonl"
    manifest_path = output_dir / "temporal_scene_narration_v2_manifest.json"

    master_records = [example.model_dump(mode="json") for example in examples]

    training_records = [example.to_training_record() for example in examples]

    temporary_paths = [
        master_path.with_suffix(".jsonl.tmp"),
        training_path.with_suffix(".jsonl.tmp"),
        manifest_path.with_suffix(".json.tmp"),
    ]

    try:
        _write_jsonl(temporary_paths[0], master_records)
        _write_jsonl(temporary_paths[1], training_records)

        if len(training_records) != validation["examples"]:
            raise ValueError("Training export count does not match validated examples.")

        manifest = {
            "dataset_name": "temporal_scene_narration",
            "dataset_version": "2.0.0",
            "generator_version": TEMPORAL_SCENE_GENERATOR_V2_VERSION,
            "total_examples": validation["examples"],
            "unique_scenarios": validation["unique_scenarios"],
            "unique_patients": validation["unique_patients"],
            "unique_example_ids": validation["unique_example_ids"],
            "unique_temporal_targets": validation["unique_temporal_targets"],
            "scenario_indices": list(scenario_indices),
            "split_status": "not_split",
            "split_reason": (
                "Dataset is too small for meaningful "
                "scenario-level train/validation/test splitting."
            ),
            "validation_status": "passed",
            "master_file": master_path.name,
            "training_file": training_path.name,
            "training_format": "messages",
        }

        _write_jsonl(temporary_paths[2], [manifest])

        for temporary_path, destination_path in zip(
            temporary_paths,
            (master_path, training_path, manifest_path),
            strict=True,
        ):
            temporary_path.replace(destination_path)

    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

    manifest["master_sha256"] = _sha256(master_path)
    manifest["training_sha256"] = _sha256(training_path)

    manifest_temporary_path = manifest_path.with_suffix(".json.tmp")

    try:
        _write_jsonl(manifest_temporary_path, [manifest])
        manifest_temporary_path.replace(manifest_path)
    finally:
        if manifest_temporary_path.exists():
            manifest_temporary_path.unlink()

    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"

    manifest = generate_and_export(output_dir)

    print("Temporal V2 dataset generation successful.")
    print(f"Total examples: {manifest['total_examples']}")
    print(f"Unique scenarios: {manifest['unique_scenarios']}")
    print(f"Unique patients: {manifest['unique_patients']}")
    print(f"Unique temporal targets: {manifest['unique_temporal_targets']}")
    print(f"Validation: {manifest['validation_status']}")
    print(f"Master dataset: {output_dir / manifest['master_file']}")
    print(f"Training dataset: {output_dir / manifest['training_file']}")
    print(f"Manifest: {output_dir / manifest_path_name()}")
    print("Scenario-level split: deferred")


def manifest_path_name() -> str:
    return "temporal_scene_narration_v2_manifest.json"


if __name__ == "__main__":
    main()
