import random

from sceneops.simulator.deterioration import DeteriorationEvent
from sceneops.simulator.patients import Patient
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
)
from sceneops.simulator.triage import (
    TriageCategory,
    classify_patient,
)

ARCHETYPE_TO_CATEGORY = {
    PatientArchetype.GREEN: TriageCategory.GREEN,
    PatientArchetype.YELLOW: TriageCategory.YELLOW,
    PatientArchetype.RED: TriageCategory.RED,
    PatientArchetype.BLACK: TriageCategory.BLACK,
    PatientArchetype.PENDING: TriageCategory.PENDING,
}


class ScenarioEventGenerator:
    """Generate reproducible deterioration events for a scenario."""

    def generate(
        self,
        config: ScenarioConfig,
        patients: list[Patient],
    ) -> list[DeteriorationEvent]:
        rng = random.Random(config.seed)

        patient_groups: dict[PatientArchetype, list[Patient]] = {
            archetype: [] for archetype in PatientArchetype
        }

        for patient in patients:
            category = classify_patient(patient).category

            for archetype, expected_category in ARCHETYPE_TO_CATEGORY.items():
                if category == expected_category:
                    patient_groups[archetype].append(patient)
                    break

        events = []

        for spec in config.event_specs:
            eligible_patients = patient_groups[spec.target_archetype]

            if not eligible_patients:
                raise ValueError(
                    f"No patients available for archetype "
                    f"{spec.target_archetype.value}."
                )

            selected_patient = rng.choice(eligible_patients)
            event_time = rng.randint(
                spec.earliest_seconds,
                spec.latest_seconds,
            )

            events.append(
                DeteriorationEvent(
                    event_id=spec.event_id,
                    patient_id=selected_patient.patient_id,
                    at_seconds=event_time,
                    changes=spec.changes.model_copy(deep=True),
                )
            )

        return events
