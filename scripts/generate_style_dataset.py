import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from sceneops.training.style_selection_dataset import (
    GENERATOR_VERSION,
    SCENARIO_COUNT,
    STYLE_LABELS,
    StyleSelectionDatasetGenerator,
)

RANDOM_SEED = 42
MINIMUM_EXAMPLES = 5500


def write_jsonl(path: Path, examples: list) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            file.write(
                json.dumps(
                    example.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def split_by_scenario(examples: list) -> dict[str, list]:
    scenario_ids = sorted({example.scenario_id for example in examples})

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(scenario_ids)

    train_end = int(len(scenario_ids) * 0.60)
    validation_end = int(len(scenario_ids) * 0.80)

    scenario_groups = {
        "train": set(scenario_ids[:train_end]),
        "validation": set(scenario_ids[train_end:validation_end]),
        "test": set(scenario_ids[validation_end:]),
    }

    return {
        name: [
            example for example in examples if example.scenario_id in selected_scenarios
        ]
        for name, selected_scenarios in scenario_groups.items()
    }


def validate_dataset(examples: list, splits: dict) -> dict:
    if len(examples) < MINIMUM_EXAMPLES:
        raise ValueError(
            f"Expected at least {MINIMUM_EXAMPLES} examples; received {len(examples)}."
        )

    example_ids = [example.example_id for example in examples]

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate example IDs detected.")

    if len({example.scenario_id for example in examples}) != SCENARIO_COUNT:
        raise ValueError("Unexpected number of scenarios.")

    style_counts = Counter(example.target_style for example in examples)

    if set(style_counts) != set(STYLE_LABELS):
        raise ValueError("One or more expected style labels are missing.")

    if any(count == 0 for count in style_counts.values()):
        raise ValueError("Empty style class detected.")

    for example in examples:
        messages = example.messages

        if len(messages) != 3:
            raise ValueError(f"Unexpected message count: {example.example_id}")

        if messages[0].role != "system":
            raise ValueError("First message must be system.")

        if messages[1].role != "user":
            raise ValueError("Second message must be user.")

        if messages[2].role != "assistant":
            raise ValueError("Third message must be assistant.")

        expected_target = json.dumps(
            {"style": example.target_style},
            ensure_ascii=False,
        )

        if messages[2].content != expected_target:
            raise ValueError(f"Invalid target format: {example.example_id}")

    split_scenarios = {
        name: {example.scenario_id for example in records}
        for name, records in splits.items()
    }

    names = ("train", "validation", "test")

    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = split_scenarios[first] & split_scenarios[second]

            if overlap:
                raise ValueError(
                    f"Scenario leakage between {first} and {second}: {sorted(overlap)}"
                )

    split_ids = [
        example.example_id for records in splits.values() for example in records
    ]

    if len(split_ids) != len(set(split_ids)):
        raise ValueError("Duplicate IDs across splits.")

    if set(split_ids) != set(example_ids):
        raise ValueError("Split files do not cover the master dataset.")

    return {
        "style_counts": dict(sorted(style_counts.items())),
        "scenario_counts": {name: len(split_scenarios[name]) for name in names},
    }


def main() -> None:
    generator = StyleSelectionDatasetGenerator()
    examples = generator.generate()

    splits = split_by_scenario(examples)
    report = validate_dataset(examples, splits)

    processed_dir = Path("data/processed")
    split_dir = Path("data/splits/style_selection")

    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "master": processed_dir / "style_selection_v1.jsonl",
        "train": split_dir / "train.jsonl",
        "validation": split_dir / "validation.jsonl",
        "test": split_dir / "test.jsonl",
    }

    staged_files = {
        name: Path(str(path) + ".tmp") for name, path in output_files.items()
    }

    manifest_path = split_dir / "manifest.json"
    staged_manifest = Path(str(manifest_path) + ".tmp")

    try:
        write_jsonl(staged_files["master"], examples)

        for name, records in splits.items():
            write_jsonl(staged_files[name], records)

        # Reload every staged file and verify record counts.
        for name, path in staged_files.items():
            with path.open("r", encoding="utf-8") as file:
                saved_count = sum(1 for line in file if line.strip())

            expected_count = len(examples) if name == "master" else len(splits[name])

            if saved_count != expected_count:
                raise ValueError(
                    f"{name} file count mismatch: {saved_count} != {expected_count}"
                )

        manifest = {
            "generator_version": GENERATOR_VERSION,
            "random_seed": RANDOM_SEED,
            "minimum_required_examples": MINIMUM_EXAMPLES,
            "total_examples": len(examples),
            "unique_scenarios": SCENARIO_COUNT,
            "style_labels": list(STYLE_LABELS),
            "style_counts": report["style_counts"],
            "split_strategy": "scenario_level",
            "splits": {
                name: {
                    "examples": len(splits[name]),
                    "scenarios": report["scenario_counts"][name],
                }
                for name in ("train", "validation", "test")
            },
            "files": {
                name: {
                    "path": str(output_files[name]).replace("\\", "/"),
                    "sha256": calculate_sha256(staged_files[name]),
                    "size_bytes": staged_files[name].stat().st_size,
                }
                for name in output_files
            },
        }

        staged_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Publish data files first and manifest last.
        for name, staged_path in staged_files.items():
            staged_path.replace(output_files[name])

        staged_manifest.replace(manifest_path)

    finally:
        for path in staged_files.values():
            path.unlink(missing_ok=True)

        staged_manifest.unlink(missing_ok=True)

    print("Style-selection dataset generation successful.")
    print("Total examples:", len(examples))
    print("Unique scenarios:", SCENARIO_COUNT)

    print("\nStyle distribution:")
    for style, count in report["style_counts"].items():
        print(f"{style}: {count}")

    print("\nSplit distribution:")
    for name, records in splits.items():
        print(
            f"{name.title()}: {len(records)} examples, "
            f"{report['scenario_counts'][name]} scenarios"
        )

    print("\nMaster dataset:", output_files["master"])
    print("Split directory:", split_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
