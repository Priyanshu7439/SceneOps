import hashlib
import json
import shutil
from pathlib import Path

from sceneops.training.dataset import (
    DEFAULT_SCENARIO_INDICES,
    GENERATOR_VERSION,
    DatasetExample,
    PatientDialogueDatasetGenerator,
)
from sceneops.training.splitting import split_by_scenario
from sceneops.training.validator import (
    PatientDialogueDatasetValidator,
)

RANDOM_SEED = 42
MINIMUM_EXAMPLES = 5500


def write_jsonl(path: Path, examples: list[DatasetExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            file.write(
                json.dumps(
                    example.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(path: Path) -> list[DatasetExample]:
    examples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record at {path}, line {line_number}.")

            try:
                examples.append(DatasetExample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid record in {path}, line {line_number}: {exc}"
                ) from exc

    return examples


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def publish_outputs(
    staged_files: dict[str, Path],
    output_files: dict[str, Path],
) -> None:
    """
    Publish validated files with rollback on replacement failure.

    Individual replacements are atomic on the same filesystem.
    The complete multi-file operation is not atomic.
    Backups are preserved if rollback itself fails.
    """
    if set(staged_files) != set(output_files):
        raise ValueError("Staged files and output files must have matching keys.")

    backups: dict[str, Path] = {}
    published: list[str] = []

    try:
        for name, output_path in output_files.items():
            if output_path.exists():
                backup_path = Path(str(output_path) + ".bak.tmp")

                if backup_path.exists():
                    raise FileExistsError(
                        f"Existing backup requires manual inspection: {backup_path}"
                    )

                shutil.copy2(output_path, backup_path)
                backups[name] = backup_path

        for name, staged_path in staged_files.items():
            staged_path.replace(output_files[name])
            published.append(name)

    except Exception as publish_error:
        rollback_errors = []

        for name in reversed(published):
            output_path = output_files[name]
            backup_path = backups.get(name)

            try:
                if backup_path is not None:
                    backup_path.replace(output_path)
                else:
                    output_path.unlink(missing_ok=True)

            except OSError as rollback_error:
                rollback_errors.append(f"{name}: {rollback_error}")

        if rollback_errors:
            raise RuntimeError(
                "Publishing failed and rollback was incomplete. "
                "Backup files have been preserved for manual recovery. "
                + "; ".join(rollback_errors)
            ) from publish_error

        raise

    else:
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)


def main() -> None:
    generator = PatientDialogueDatasetGenerator()
    validator = PatientDialogueDatasetValidator()

    processed_dir = Path("data/processed")
    splits_dir = Path("data/splits")

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    master_file = processed_dir / "patient_dialogue_v2.jsonl"
    manifest_file = splits_dir / "split_manifest.json"

    output_files = {
        "master": master_file,
        "train": splits_dir / "train.jsonl",
        "validation": splits_dir / "validation.jsonl",
        "test": splits_dir / "test.jsonl",
    }

    temporary_files = {
        name: Path(str(path) + ".tmp") for name, path in output_files.items()
    }

    temporary_manifest = Path(str(manifest_file) + ".tmp")

    staged_files = {
        **temporary_files,
        "manifest": temporary_manifest,
    }

    final_files = {
        **output_files,
        "manifest": manifest_file,
    }

    try:
        # Step 1: Generate and validate the complete dataset.
        examples = generator.generate()
        generation_report = validator.validate(examples)

        if len(examples) < MINIMUM_EXAMPLES:
            raise ValueError(
                f"Dataset has {len(examples)} examples. "
                f"Minimum required: {MINIMUM_EXAMPLES}."
            )

        if not examples:
            raise ValueError("Generated dataset cannot be empty.")

        # Step 2: Create scenario-level splits.
        splits = split_by_scenario(
            examples,
            seed=RANDOM_SEED,
        )

        # Step 3: Write datasets to temporary files.
        write_jsonl(temporary_files["master"], examples)

        for split_name, split_examples in splits.items():
            write_jsonl(
                temporary_files[split_name],
                split_examples,
            )

        # Step 4: Reload and validate every saved JSONL.
        reloaded = {}

        for name, temporary_path in temporary_files.items():
            saved_examples = load_jsonl(temporary_path)
            validator.validate(saved_examples)
            reloaded[name] = saved_examples

        # Step 5: Verify scenario isolation after disk reload.
        scenario_sets = {
            name: {example.scenario_id for example in reloaded[name]}
            for name in ("train", "validation", "test")
        }

        if (
            scenario_sets["train"] & scenario_sets["validation"]
            or scenario_sets["train"] & scenario_sets["test"]
            or scenario_sets["validation"] & scenario_sets["test"]
        ):
            raise ValueError("Scenario leakage detected in saved split files.")

        # Step 6: Verify no examples were lost or duplicated.
        original_ids = {example.example_id for example in reloaded["master"]}

        split_ids = [
            example.example_id
            for name in ("train", "validation", "test")
            for example in reloaded[name]
        ]

        if len(split_ids) != len(set(split_ids)):
            raise ValueError("Duplicate example IDs found across saved splits.")

        if set(split_ids) != original_ids:
            raise ValueError("Saved splits do not contain exactly the master examples.")

        # Step 7: Build reproducibility manifest with checksums.
        file_metadata = {
            name: {
                "path": str(output_files[name]).replace("\\", "/"),
                "sha256": calculate_sha256(temporary_files[name]),
                "size_bytes": temporary_files[name].stat().st_size,
            }
            for name in output_files
        }

        scenario_union = (
            scenario_sets["train"] | scenario_sets["validation"] | scenario_sets["test"]
        )

        manifest = {
            "generator_version": GENERATOR_VERSION,
            "random_seed": RANDOM_SEED,
            "scenario_indices": list(DEFAULT_SCENARIO_INDICES),
            "split_strategy": "scenario_level",
            "minimum_required_examples": MINIMUM_EXAMPLES,
            "total_examples": len(examples),
            "unique_scenarios": len(scenario_union),
            "validation_report": generation_report,
            "splits": {
                name: {
                    "examples": len(reloaded[name]),
                    "scenarios": sorted(scenario_sets[name]),
                }
                for name in ("train", "validation", "test")
            },
            "files": file_metadata,
        }

        temporary_manifest.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        # Step 8: Publish only after all validations succeed.
        publish_outputs(staged_files, final_files)

    finally:
        for temporary_path in staged_files.values():
            temporary_path.unlink(missing_ok=True)

    print("Dataset generation, splitting and validation successful.")
    print(f"Total examples: {generation_report['examples']}")
    print(f"Unique scenarios: {len(scenario_union)}")

    for name in ("train", "validation", "test"):
        print(
            f"{name.title()}: "
            f"{len(reloaded[name])} examples, "
            f"{len(scenario_sets[name])} scenarios"
        )

    print(f"Master dataset: {master_file}")
    print(f"Split directory: {splits_dir}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
