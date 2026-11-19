from collections import defaultdict
from itertools import combinations

from sceneops.simulator.scenario_factory import create_hybrid_scenario
from sceneops.simulator.triage import classify_patient
from sceneops.training.scenario_diversity_evaluator import (
    ScenarioDiversityEvaluator,
)

PROFILE_GROUPS = {
    "clinical_demographic": (
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
    ),
    "communication_metadata": (
        "personality",
        "speech_style",
    ),
    "pain_state": ("pain_level",),
}


class ScenarioSimilarityEvaluator:
    """Measure diversity across patient profile dimensions."""

    def _compare_profiles(self, profiles: list[tuple]) -> dict:
        similarities = []
        exact_duplicate_pairs = 0

        for first, second in combinations(profiles, 2):
            matching_fields = sum(left == right for left, right in zip(first, second))
            similarity = matching_fields / len(first)
            similarities.append(similarity)

            if similarity == 1.0:
                exact_duplicate_pairs += 1

        pair_count = len(similarities)

        return {
            "profiles": len(profiles),
            "unique_profiles": len(set(profiles)),
            "pairs_compared": pair_count,
            "average_similarity": (
                sum(similarities) / pair_count if pair_count else 0.0
            ),
            "maximum_similarity": (max(similarities) if pair_count else 0.0),
            "exact_duplicate_pairs": exact_duplicate_pairs,
        }

    def evaluate(self, seed_count: int = 100) -> dict:
        if isinstance(seed_count, bool) or not isinstance(seed_count, int):
            raise TypeError("seed_count must be an integer.")

        if seed_count < 1:
            raise ValueError("seed_count must be >= 1.")

        groups = defaultdict(lambda: defaultdict(list))
        config_builder = ScenarioDiversityEvaluator()

        for seed in range(seed_count):
            config = config_builder._create_config(seed)
            scenario = create_hybrid_scenario(config)

            for patient in scenario.patients:
                category = classify_patient(patient).category.value
                profile = patient.model_dump(mode="json")

                for group_name, fields in PROFILE_GROUPS.items():
                    normalized_profile = tuple(profile[field] for field in fields)
                    groups[category][group_name].append(normalized_profile)

        results = {}

        for category, category_groups in sorted(groups.items()):
            results[category] = {}

            for group_name, profiles in sorted(category_groups.items()):
                results[category][group_name] = self._compare_profiles(profiles)

        return results
