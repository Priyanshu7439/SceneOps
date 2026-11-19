import random

from sceneops.simulator.patients import (
    InjuryType,
    MentalStatus,
    Patient,
)
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
)
from sceneops.simulator.triage import (
    TriageCategory,
    classify_patient,
)

EXPECTED_TRIAGE_CATEGORY = {
    PatientArchetype.GREEN: TriageCategory.GREEN,
    PatientArchetype.YELLOW: TriageCategory.YELLOW,
    PatientArchetype.RED: TriageCategory.RED,
    PatientArchetype.BLACK: TriageCategory.BLACK,
    PatientArchetype.PENDING: TriageCategory.PENDING,
}

PERSONALITIES = (
    "calm",
    "anxious",
    "cooperative",
    "irritable",
    "confused",
)

SPEECH_STYLES = (
    "plain",
    "brief",
    "hesitant",
    "descriptive",
    "repetitive",
)

NORMAL_REFILL_TIMES = (0.8, 1.2, 1.8)
RED_REFILL_TIMES = (1.0, 1.5, 2.1, 2.8, 3.5)

RESPONSIVE_STATES = (
    (MentalStatus.ALERT, True),
    (MentalStatus.CONFUSED, True),
)

RED_MENTAL_STATES = (
    (MentalStatus.ALERT, True),
    (MentalStatus.CONFUSED, True),
    (MentalStatus.UNRESPONSIVE, False),
)


class PatientProfileGenerator:
    """Generate reproducible synthetic patient profiles."""

    def generate(self, config: ScenarioConfig) -> list[Patient]:
        rng = random.Random(config.seed)

        archetypes = [
            archetype
            for archetype, count in config.archetype_counts.items()
            for _ in range(count)
        ]

        rng.shuffle(archetypes)

        patients = []

        for index, archetype in enumerate(archetypes, start=1):
            patient = self._create_patient(
                patient_id=f"P{index:03d}",
                archetype=archetype,
                rng=rng,
            )

            actual_category = classify_patient(patient).category
            expected_category = EXPECTED_TRIAGE_CATEGORY[archetype]

            if actual_category != expected_category:
                raise ValueError(
                    f"Patient {patient.patient_id} was configured as "
                    f"{archetype.value} but classified as "
                    f"{actual_category.value}."
                )

            patients.append(patient)

        return patients

    @staticmethod
    def _create_patient(
        patient_id: str,
        archetype: PatientArchetype,
        rng: random.Random,
    ) -> Patient:
        common_attributes = {
            "patient_id": patient_id,
            "age": rng.randint(18, 75),
            "injury_type": rng.choice(list(InjuryType)),
            "major_hemorrhage": False,
            "personality": rng.choice(PERSONALITIES),
            "speech_style": rng.choice(SPEECH_STYLES),
            "pain_level": rng.randint(0, 10),
        }

        profiles = {
            PatientArchetype.GREEN: {
                "ambulatory": True,
                "respiratory_rate": rng.choice((18, 20, 22, 24)),
                "radial_pulse_present": True,
                "follows_commands": True,
                "mental_status": rng.choice(RESPONSIVE_STATES)[0],
                "capillary_refill_seconds": rng.choice(NORMAL_REFILL_TIMES),
            },
            PatientArchetype.YELLOW: {
                "ambulatory": False,
                "respiratory_rate": rng.randint(18, 24),
                "radial_pulse_present": True,
                "follows_commands": True,
                "mental_status": rng.choice(RESPONSIVE_STATES)[0],
                "capillary_refill_seconds": rng.choice(NORMAL_REFILL_TIMES),
            },
            PatientArchetype.RED: {
                "ambulatory": False,
                "respiratory_rate": rng.randint(31, 40),
                "radial_pulse_present": rng.choice((True, False)),
                "follows_commands": True,
                "mental_status": MentalStatus.ALERT,
                "capillary_refill_seconds": rng.choice(RED_REFILL_TIMES),
                "major_hemorrhage": rng.choice((True, False)),
            },
            PatientArchetype.BLACK: {
                "ambulatory": False,
                "respiratory_rate": 0,
                "airway_opened": True,
                "radial_pulse_present": False,
                "follows_commands": False,
                "mental_status": MentalStatus.UNRESPONSIVE,
            },
            PatientArchetype.PENDING: {
                "ambulatory": False,
                "respiratory_rate": None,
                "radial_pulse_present": True,
                "follows_commands": True,
                "mental_status": rng.choice(RESPONSIVE_STATES)[0],
            },
        }

        if archetype == PatientArchetype.RED:
            mental_status, follows_commands = rng.choice(RED_MENTAL_STATES)
            profiles[archetype]["mental_status"] = mental_status
            profiles[archetype]["follows_commands"] = follows_commands

        patient_attributes = {
            **common_attributes,
            **profiles[archetype],
        }

        return Patient(**patient_attributes)
