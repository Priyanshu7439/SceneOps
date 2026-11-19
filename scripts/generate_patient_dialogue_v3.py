from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sceneops.training.patient_dialogue_dataset_v3 import (
    DEFAULT_SCENARIO_INDICES,
    PatientDialogueDatasetGeneratorV3,
)
from sceneops.training.patient_dialogue_validator_v3 import (
    PatientDialogueDatasetValidatorV3,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits" / "v3"

FULL_DATASET_PATH = PROCESSED_DIR / "patient_dialogue_v3.jsonl"

RANDOM_SEED = 42

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

EXPECTED_PATIENTS_PER_SCENARIO = 20
EXPECTED_EXAMPLES_PER_PATIENT = 11


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def split_scenarios(
    scenario_ids: list[str],
) -> dict[str, list[str]]:
    """Create deterministic scenario-level 60/20/20 splits."""

    shuffled = sorted(scenario_ids)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    total = len(shuffled)

    train_count = int(total * TRAIN_RATIO)
    validation_count = int(total * VALIDATION_RATIO)

    train_scenarios = shuffled[:train_count]

    validation_scenarios = shuffled[train_count : train_count + validation_count]

    test_scenarios = shuffled[train_count + validation_count :]

    return {
        "train": sorted(train_scenarios),
        "validation": sorted(validation_scenarios),
        "test": sorted(test_scenarios),
    }


def verify_scenario_distribution(
    examples: list,
    scenario_splits: dict[str, list[str]],
) -> None:
    """Verify expected patient and example counts."""

    scenario_counts = Counter(example.scenario_id for example in examples)

    patient_counts = defaultdict(set)

    for example in examples:
        patient_counts[example.scenario_id].add(example.patient_id)

    for scenario_id in DEFAULT_SCENARIO_INDICES:
        expected_id = f"MCI-{scenario_id:03d}"

        actual_examples = scenario_counts.get(expected_id, 0)
        actual_patients = len(patient_counts[expected_id])

        expected_examples = (
            EXPECTED_PATIENTS_PER_SCENARIO * EXPECTED_EXAMPLES_PER_PATIENT
        )

        if actual_patients != EXPECTED_PATIENTS_PER_SCENARIO:
            raise ValueError(
                f"{expected_id}: expected "
                f"{EXPECTED_PATIENTS_PER_SCENARIO} patients, "
                f"found {actual_patients}."
            )

        if actual_examples != expected_examples:
            raise ValueError(
                f"{expected_id}: expected "
                f"{expected_examples} examples, "
                f"found {actual_examples}."
            )

    # Ensure every scenario belongs to exactly one split.
    all_split_scenarios = []

    for scenarios in scenario_splits.values():
        all_split_scenarios.extend(scenarios)

    if len(all_split_scenarios) != len(set(all_split_scenarios)):
        raise ValueError(
            "Scenario leakage detected: a scenario appears in multiple splits."
        )

    expected_scenarios = {f"MCI-{index:03d}" for index in DEFAULT_SCENARIO_INDICES}

    if set(all_split_scenarios) != expected_scenarios:
        raise ValueError(
            "Scenario split does not contain exactly the expected 92 scenarios."
        )


def build_metadata_record(example) -> dict:
    return {
        "example_id": example.example_id,
        "scenario_id": example.scenario_id,
        "patient_id": example.patient_id,
        "generator_version": example.generator_version,
        "category": example.category,
        "authorized_facts": list(example.authorized_facts),
        "speech_style": example.speech_style,
    }


def main() -> None:
    print("=" * 65)
    print("SCENEOPS - PATIENT DIALOGUE DATASET V3")
    print("=" * 65)

    print("\n[1/6] Generating synthetic patient dialogue examples...")

    generator = PatientDialogueDatasetGeneratorV3()

    examples = generator.generate(scenario_indices=DEFAULT_SCENARIO_INDICES)

    print(f"Generated examples: {len(examples)}")

    print("\n[2/6] Running V3 dataset validation...")

    validator = PatientDialogueDatasetValidatorV3()
    validation_report = validator.validate(examples)

    print(
        json.dumps(
            validation_report.summary(),
            indent=2,
        )
    )

    if validation_report.warnings:
        print("\nValidation warnings:")

        for warning in validation_report.warnings:
            print(f"- {warning}")

    if validation_report.errors:
        print("\nValidation errors:")

        for error in validation_report.errors:
            print(f"- {error}")

    validation_report.raise_for_errors()

    print("\nV3 validation passed.")

    print("\n[3/6] Creating scenario-level splits...")

    scenario_ids = sorted({example.scenario_id for example in examples})

    scenario_splits = split_scenarios(scenario_ids)

    verify_scenario_distribution(
        examples,
        scenario_splits,
    )

    split_summary = {}

    for split_name, split_scenario_ids in scenario_splits.items():
        split_examples = [
            example
            for example in examples
            if example.scenario_id in set(split_scenario_ids)
        ]

        split_summary[split_name] = {
            "scenario_count": len(split_scenario_ids),
            "example_count": len(split_examples),
            "patient_count": len(
                {
                    (example.scenario_id, example.patient_id)
                    for example in split_examples
                }
            ),
        }

        print(
            f"{split_name}: "
            f"{len(split_scenario_ids)} scenarios, "
            f"{len(split_examples)} examples"
        )

    print("\n[4/6] Preparing output files...")

    full_records = [example.model_dump(mode="json") for example in examples]

    split_records = {}

    for split_name, split_scenario_ids in scenario_splits.items():
        selected_ids = set(split_scenario_ids)

        selected_examples = [
            example for example in examples if example.scenario_id in selected_ids
        ]

        # TRL-compatible conversational records.
        training_records = [
            example.to_training_record() for example in selected_examples
        ]

        # Metadata is stored separately for traceability
        # and evaluation without changing the SFT messages.
        metadata_records = [
            build_metadata_record(example) for example in selected_examples
        ]

        split_records[split_name] = {
            "training": training_records,
            "metadata": metadata_records,
        }

    manifest = {
        "dataset_name": "SceneOps Patient Dialogue V3",
        "generator_version": "3.1.0",
        "random_seed": RANDOM_SEED,
        "total_examples": len(examples),
        "total_scenarios": len(scenario_ids),
        "total_patients": len(
            {(example.scenario_id, example.patient_id) for example in examples}
        ),
        "category_counts": dict(Counter(example.category for example in examples)),
        "speech_style_counts": dict(
            Counter(example.speech_style for example in examples)
        ),
        "scenario_split_method": "deterministic scenario-level split",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": round(
                1.0 - TRAIN_RATIO - VALIDATION_RATIO,
                2,
            ),
        },
        "splits": {
            split_name: {
                **split_summary[split_name],
                "scenario_ids": scenario_splits[split_name],
            }
            for split_name in scenario_splits
        },
        "validation": validation_report.summary(),
    }

    print("\n[5/6] Writing staged dataset files...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".patient_dialogue_v3_staging_",
            dir=DATA_DIR,
        )
    )

    try:
        staged_full_path = staging_dir / "patient_dialogue_v3.jsonl"

        write_jsonl(
            staged_full_path,
            full_records,
        )

        staged_split_dir = staging_dir / "splits"

        for split_name, records in split_records.items():
            write_jsonl(
                staged_split_dir / f"{split_name}.jsonl",
                records["training"],
            )

            write_jsonl(
                staged_split_dir / f"{split_name}_metadata.jsonl",
                records["metadata"],
            )

        manifest_path = staged_split_dir / "split_manifest.json"

        manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n[6/6] Publishing generated dataset...")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)

        shutil.move(
            str(staged_full_path),
            str(FULL_DATASET_PATH),
        )

        for file_path in staged_split_dir.iterdir():
            destination = SPLIT_DIR / file_path.name

            shutil.move(
                str(file_path),
                str(destination),
            )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    print("\n" + "=" * 65)
    print("PATIENT DIALOGUE V3 GENERATION SUCCESSFUL")
    print("=" * 65)

    print(f"\nFull dataset: {FULL_DATASET_PATH}")
    print(f"Split directory: {SPLIT_DIR}")

    print("\nFinal split summary:")

    for split_name, summary in split_summary.items():
        print(
            f"{split_name}: "
            f"{summary['scenario_count']} scenarios, "
            f"{summary['patient_count']} patients, "
            f"{summary['example_count']} examples"
        )

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
