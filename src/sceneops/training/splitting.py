from __future__ import annotations

import random

from sceneops.training.dataset import DatasetExample

SPLIT_NAMES = ("train", "validation", "test")


def split_by_scenario(
    examples: list[DatasetExample],
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[DatasetExample]]:
    """
    Split examples by scenario, never by individual example.

    Every example belonging to one scenario is assigned to the
    same split. The remaining ratio is assigned to the test split.
    """
    if not examples:
        raise ValueError("Cannot split an empty dataset.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    scenarios: dict[str, list[DatasetExample]] = {}

    for example in examples:
        scenarios.setdefault(example.scenario_id, []).append(example)

    scenario_ids = sorted(scenarios)

    if len(scenario_ids) < 3:
        raise ValueError(
            "At least 3 unique scenarios are required for "
            "train/validation/test splitting."
        )

    rng = random.Random(seed)
    rng.shuffle(scenario_ids)

    scenario_count = len(scenario_ids)

    train_count = max(1, int(scenario_count * train_ratio))
    validation_count = max(1, int(scenario_count * validation_ratio))

    # Keep at least one scenario for the test split.
    if train_count + validation_count >= scenario_count:
        raise ValueError("Not enough scenarios for the requested split ratios.")

    train_scenarios = scenario_ids[:train_count]
    validation_scenarios = scenario_ids[train_count : train_count + validation_count]
    test_scenarios = scenario_ids[train_count + validation_count :]

    split_scenarios = {
        "train": train_scenarios,
        "validation": validation_scenarios,
        "test": test_scenarios,
    }

    splits = {
        split_name: [
            example
            for scenario_id in assigned_scenarios
            for example in scenarios[scenario_id]
        ]
        for split_name, assigned_scenarios in split_scenarios.items()
    }

    # Defensive checks: no scenario leakage and no lost examples.
    scenario_sets = {
        name: set(assigned_scenarios)
        for name, assigned_scenarios in split_scenarios.items()
    }

    if (
        scenario_sets["train"] & scenario_sets["validation"]
        or scenario_sets["train"] & scenario_sets["test"]
        or scenario_sets["validation"] & scenario_sets["test"]
    ):
        raise RuntimeError("Scenario leakage detected between splits.")

    original_ids = {example.example_id for example in examples}
    split_ids = [example.example_id for split in splits.values() for example in split]

    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError("Duplicate examples found across splits.")

    if set(split_ids) != original_ids:
        raise RuntimeError("Examples were lost or added during splitting.")

    return splits
