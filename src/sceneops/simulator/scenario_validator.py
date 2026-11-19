from collections import Counter

from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_config import (
    ScenarioConfig,
)
from sceneops.simulator.scenario_event_generator import (
    ARCHETYPE_TO_CATEGORY,
)
from sceneops.simulator.triage import classify_patient


class ScenarioValidator:
    """Validate a generated scenario against its configuration."""

    def validate(
        self,
        config: ScenarioConfig,
        scenario: ScenarioDefinition,
    ) -> None:
        self._validate_metadata(config, scenario)
        self._validate_patient_count(config, scenario)
        self._validate_ambulance_count(config, scenario)
        self._validate_patient_ids(scenario)
        self._validate_archetype_distribution(config, scenario)
        self._validate_events(config, scenario)

    @staticmethod
    def _validate_metadata(
        config: ScenarioConfig,
        scenario: ScenarioDefinition,
    ) -> None:
        if scenario.scenario_id != config.scenario_id:
            raise ValueError("Scenario ID does not match configuration.")

        if scenario.duration_seconds != config.duration_seconds:
            raise ValueError("Scenario duration does not match configuration.")

    @staticmethod
    def _validate_patient_count(
        config: ScenarioConfig,
        scenario: ScenarioDefinition,
    ) -> None:
        if len(scenario.patients) != config.patient_count:
            raise ValueError("Generated patient count does not match configuration.")

    @staticmethod
    def _validate_ambulance_count(
        config: ScenarioConfig,
        scenario: ScenarioDefinition,
    ) -> None:
        if len(scenario.ambulances) != config.ambulance_count:
            raise ValueError("Generated ambulance count does not match configuration.")

    @staticmethod
    def _validate_patient_ids(
        scenario: ScenarioDefinition,
    ) -> None:
        patient_ids = [patient.patient_id for patient in scenario.patients]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Patient IDs must be unique.")

        ambulance_ids = [ambulance.ambulance_id for ambulance in scenario.ambulances]

        if len(ambulance_ids) != len(set(ambulance_ids)):
            raise ValueError("Ambulance IDs must be unique.")

    @staticmethod
    def _validate_archetype_distribution(
        config: ScenarioConfig,
        scenario: ScenarioDefinition,
    ) -> None:
        actual_distribution = Counter(
            classify_patient(patient).category for patient in scenario.patients
        )

        expected_distribution = {
            ARCHETYPE_TO_CATEGORY[archetype]: count
            for archetype, count in config.archetype_counts.items()
        }

        if actual_distribution != expected_distribution:
            raise ValueError(
                "Generated patient triage distribution does not "
                "match configured archetype counts."
            )

    @staticmethod
    def _validate_events(
        config: ScenarioConfig,
        scenario: ScenarioDefinition,
    ) -> None:
        actual_events = {
            event.event_id: event for event in scenario.deterioration_events
        }

        if len(actual_events) != len(scenario.deterioration_events):
            raise ValueError("Deterioration event IDs must be unique.")

        expected_specs = {spec.event_id: spec for spec in config.event_specs}

        if set(actual_events) != set(expected_specs):
            raise ValueError("Generated event IDs do not match configuration.")

        patients_by_id = {patient.patient_id: patient for patient in scenario.patients}

        for event_id, spec in expected_specs.items():
            event = actual_events[event_id]

            if event.patient_id not in patients_by_id:
                raise ValueError(f"Event {event_id} references an unknown patient.")

            if not (spec.earliest_seconds <= event.at_seconds <= spec.latest_seconds):
                raise ValueError(
                    f"Event {event_id} is outside its configured time window."
                )

            if event.at_seconds > scenario.duration_seconds:
                raise ValueError(f"Event {event_id} exceeds scenario duration.")

            patient = patients_by_id[event.patient_id]
            actual_category = classify_patient(patient).category
            expected_category = ARCHETYPE_TO_CATEGORY[spec.target_archetype]

            if actual_category != expected_category:
                raise ValueError(
                    f"Event {event_id} targets a patient outside "
                    f"the {spec.target_archetype.value} archetype."
                )

            expected_changes = spec.changes.model_dump(exclude_unset=True)
            actual_changes = event.changes.model_dump(exclude_unset=True)

            if actual_changes != expected_changes:
                raise ValueError(
                    f"Event {event_id} changes do not match configuration."
                )
