from sceneops.training.patient_field_diversity_evaluator import (
    PatientFieldDiversityEvaluator,
)


def main():
    results = PatientFieldDiversityEvaluator().evaluate(seed_count=100)

    print("\nPATIENT FIELD-LEVEL DIVERSITY REPORT")
    print("=" * 60)

    for field, metrics in results.items():
        print(f"\n{field.upper()}")
        print(f"Total values: {metrics['total_values']}")
        print(f"Unique values: {metrics['unique_values']}")
        print(f"Uniqueness ratio: {metrics['uniqueness_ratio']:.2%}")
        print("Distribution:")

        for value, count in metrics["distribution"].items():
            print(f"  {value}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
