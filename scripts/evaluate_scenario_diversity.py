from sceneops.training.scenario_diversity_evaluator import (
    ScenarioDiversityEvaluator,
)


def main():
    report = ScenarioDiversityEvaluator().evaluate(seed_count=100)

    print("\nSCENARIO DIVERSITY EVALUATION")
    print("=" * 40)

    for field, value in report.__dict__.items():
        print(f"{field}: {value}")

    print(f"scenario_uniqueness_ratio: {report.scenario_uniqueness_ratio:.2%}")

    print("=" * 40)


if __name__ == "__main__":
    main()
