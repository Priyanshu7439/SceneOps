from collections import defaultdict

from sceneops.simulator.scenario_factory import create_hybrid_scenario
from sceneops.training.scenario_diversity_evaluator import (
    ScenarioDiversityEvaluator,
)

PROFILE_FIELDS = (
    "age",
    "injury_type",
    "mental_status",
    "ambulatory",
    "respiratory_rate",
    "radial_pulse_present",
    "follows_commands",
    "major_hemorrhage",
    "airway_opened",
    "capillary_refill_seconds",
    "personality",
    "speech_style",
    "pain_level",
)


class PatientFieldDiversityEvaluator:
    """Measure variation in individual patient profile fields."""

    def evaluate(self, seed_count: int = 100) -> dict:
        if isinstance(seed_count, bool) or not isinstance(seed_count, int):
            raise TypeError("seed_count must be an integer.")

        if seed_count < 1:
            raise ValueError("seed_count must be >= 1.")

        field_values = defaultdict(list)
        config_builder = ScenarioDiversityEvaluator()

        for seed in range(seed_count):
            config = config_builder._create_config(seed)
            scenario = create_hybrid_scenario(config)

            for patient in scenario.patients:
                profile = patient.model_dump(mode="json")

                for field in PROFILE_FIELDS:
                    field_values[field].append(profile[field])

        results = {}

        for field in PROFILE_FIELDS:
            values = field_values[field]
            unique_values = set(values)
            total_values = len(values)

            results[field] = {
                "total_values": total_values,
                "unique_values": len(unique_values),
                "uniqueness_ratio": (
                    len(unique_values) / total_values if total_values else 0.0
                ),
                "distribution": {
                    str(value): values.count(value)
                    for value in sorted(unique_values, key=str)
                },
            }

        return results
