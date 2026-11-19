import hashlib
import json
from dataclasses import dataclass

from sceneops.simulator.deterioration import PatientChanges
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
    ScenarioEventSpec,
)
from sceneops.simulator.scenario_factory import create_hybrid_scenario


@dataclass(frozen=True)
class DiversityReport:
    scenarios_generated: int
    unique_scenarios: int
    unique_patient_profiles: int
    unique_ages: int
    unique_injury_types: int
    unique_personalities: int
    unique_speech_styles: int
    unique_pain_levels: int
    unique_event_times: int
    unique_event_target_profiles: int

    @property
    def scenario_uniqueness_ratio(self) -> float:
        if self.scenarios_generated == 0:
            return 0.0

        return self.unique_scenarios / self.scenarios_generated


class ScenarioDiversityEvaluator:
    """Measure variation across seeded hybrid scenarios."""

    def evaluate(
        self,
        seed_count: int = 100,
    ) -> DiversityReport:
        if isinstance(seed_count, bool) or not isinstance(seed_count, int):
            raise TypeError("seed_count must be an integer.")

        if seed_count < 1:
            raise ValueError("seed_count must be >= 1.")

        scenario_hashes = set()
        patient_profile_hashes = set()

        ages = set()
        injury_types = set()
        personalities = set()
        speech_styles = set()
        pain_levels = set()
        event_times = set()
        event_target_profiles = set()

        for seed in range(seed_count):
            config = self._create_config(seed)
            scenario = create_hybrid_scenario(config)

            scenario_data = {
                "patients": [
                    patient.model_dump(mode="json") for patient in scenario.patients
                ],
                "events": [
                    event.model_dump(mode="json")
                    for event in scenario.deterioration_events
                ],
            }

            scenario_hash = self._fingerprint(scenario_data)
            scenario_hashes.add(scenario_hash)

            patients_by_id = {
                patient.patient_id: patient for patient in scenario.patients
            }

            for patient in scenario.patients:
                patient_data = patient.model_dump(mode="json")
                patient_data.pop("patient_id", None)

                patient_profile_hashes.add(self._fingerprint(patient_data))

                ages.add(patient.age)
                injury_types.add(patient.injury_type.value)
                personalities.add(patient.personality)
                speech_styles.add(patient.speech_style)
                pain_levels.add(patient.pain_level)

            for event in scenario.deterioration_events:
                event_times.add(event.at_seconds)

                target_patient = patients_by_id[event.patient_id]

                target_profile = target_patient.model_dump(mode="json")
                target_profile.pop("patient_id", None)

                event_target_profiles.add(self._fingerprint(target_profile))

        return DiversityReport(
            scenarios_generated=seed_count,
            unique_scenarios=len(scenario_hashes),
            unique_patient_profiles=len(patient_profile_hashes),
            unique_ages=len(ages),
            unique_injury_types=len(injury_types),
            unique_personalities=len(personalities),
            unique_speech_styles=len(speech_styles),
            unique_pain_levels=len(pain_levels),
            unique_event_times=len(event_times),
            unique_event_target_profiles=len(event_target_profiles),
        )

    @staticmethod
    def _create_config(seed: int) -> ScenarioConfig:
        event_specs = (
            ScenarioEventSpec(
                event_id="EV001",
                target_archetype=PatientArchetype.RED,
                earliest_seconds=100,
                latest_seconds=200,
                changes=PatientChanges(respiratory_rate=40),
            ),
            ScenarioEventSpec(
                event_id="EV002",
                target_archetype=PatientArchetype.YELLOW,
                earliest_seconds=200,
                latest_seconds=300,
                changes=PatientChanges(follows_commands=False),
            ),
        )

        return ScenarioConfig(
            scenario_id=f"MCI-DIV-{seed:03d}",
            seed=seed,
            event_specs=event_specs,
        )

    @staticmethod
    def _fingerprint(data: dict) -> str:
        serialized = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
