from sceneops.training.scenario_similarity_evaluator import (
    ScenarioSimilarityEvaluator,
)


def main():
    results = ScenarioSimilarityEvaluator().evaluate(seed_count=100)

    print("\nWITHIN-ARCHETYPE SIMILARITY ANALYSIS")
    print("=" * 55)

    for category, metrics in results.items():
        print(f"\n{category.upper()}")

        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
