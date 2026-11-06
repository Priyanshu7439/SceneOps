
import json
from pathlib import Path

from sceneops.training.dataset import (
    DatasetExample,
    PatientDialogueDatasetGenerator,
)
from sceneops.training.validator import (
    PatientDialogueDatasetValidator,
)


def main() -> None:
    generator = PatientDialogueDatasetGenerator()
    validator = PatientDialogueDatasetValidator()

    # Step 1: Generate examples.
    examples = generator.generate()

    # Step 2: Validate before writing.
    generation_report = validator.validate(examples)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "patient_dialogue_v1.jsonl"
    temporary_file = output_dir / "patient_dialogue_v1.tmp"

    try:
        # Step 3: Write to a temporary file.
        with temporary_file.open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(
                    json.dumps(
                        example.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Step 4: Reload and validate the actual saved records.
        saved_examples: list[DatasetExample] = []

        with temporary_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty JSONL record at line {line_number}."
                    )

                try:
                    saved_examples.append(
                        DatasetExample.model_validate_json(line)
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Invalid JSONL record at line {line_number}: {exc}"
                    ) from exc

        saved_report = validator.validate(saved_examples)

        # Step 5: Replace the output only after both validations pass.
        temporary_file.replace(output_file)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise

    print("Dataset generation and validation successful.")
    print(f"Generated examples: {generation_report['examples']}")
    print(f"Unique patients: {generation_report['unique_patients']}")
    print(f"Unique scenarios: {generation_report['unique_scenarios']}")
    print(f"Saved records validated: {saved_report['examples']}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
