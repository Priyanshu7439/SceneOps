from collections import Counter, defaultdict

from sceneops.simulator.patient_profile_generator import (
    PatientProfileGenerator,
)
from sceneops.simulator.patients import MentalStatus
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
)
from sceneops.simulator.triage import (
    TriageCategory,
    classify_patient,
)


def test_generates_configured_number_of_patients():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-001",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    assert len(patients) == 20


def test_generates_unique_patient_ids():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-002",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)
    patient_ids = [patient.patient_id for patient in patients]

    assert len(patient_ids) == len(set(patient_ids))


def test_preserves_configured_triage_distribution():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-003",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    categories = Counter(classify_patient(patient).category for patient in patients)

    assert categories == {
        TriageCategory.GREEN: 4,
        TriageCategory.YELLOW: 4,
        TriageCategory.RED: 4,
        TriageCategory.BLACK: 4,
        TriageCategory.PENDING: 4,
    }


def test_same_seed_produces_same_patients():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-004",
        seed=123,
    )

    generator = PatientProfileGenerator()

    first = generator.generate(config)
    second = generator.generate(config)

    assert [p.model_dump() for p in first] == [p.model_dump() for p in second]


def test_different_seeds_produce_variations():
    first_config = ScenarioConfig(
        scenario_id="MCI-TEST-005",
        seed=42,
    )
    second_config = ScenarioConfig(
        scenario_id="MCI-TEST-006",
        seed=99,
    )

    generator = PatientProfileGenerator()

    first = generator.generate(first_config)
    second = generator.generate(second_config)

    assert [p.model_dump() for p in first] != [p.model_dump() for p in second]


def test_patient_ids_follow_expected_format():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-007",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    assert patients[0].patient_id == "P001"
    assert patients[-1].patient_id == "P020"


def test_supports_custom_archetype_distribution():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-008",
        seed=42,
        patient_count=10,
        archetype_counts={
            PatientArchetype.GREEN: 2,
            PatientArchetype.YELLOW: 2,
            PatientArchetype.RED: 2,
            PatientArchetype.BLACK: 2,
            PatientArchetype.PENDING: 2,
        },
    )

    patients = PatientProfileGenerator().generate(config)

    assert len(patients) == 10


def test_unresponsive_patients_cannot_follow_commands():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-009",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    for patient in patients:
        if patient.mental_status == MentalStatus.UNRESPONSIVE:
            assert patient.follows_commands is False


def test_red_patients_preserve_red_classification():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-010",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    red_patients = [
        patient
        for patient in patients
        if classify_patient(patient).category == TriageCategory.RED
    ]

    assert len(red_patients) == 4

    for patient in red_patients:
        assert patient.respiratory_rate > 30


def test_normal_refill_values_preserve_non_red_categories():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-011",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    for patient in patients:
        category = classify_patient(patient).category

        if category in (TriageCategory.GREEN, TriageCategory.YELLOW):
            assert patient.capillary_refill_seconds <= 2


def test_black_patients_preserve_apnea_state():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-012",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    black_patients = [
        patient
        for patient in patients
        if classify_patient(patient).category == TriageCategory.BLACK
    ]

    assert len(black_patients) == 4

    for patient in black_patients:
        assert patient.respiratory_rate == 0
        assert patient.airway_opened is True
        assert patient.mental_status == MentalStatus.UNRESPONSIVE
        assert patient.follows_commands is False


def test_pending_patients_preserve_missing_respiratory_assessment():
    config = ScenarioConfig(
        scenario_id="MCI-TEST-013",
        seed=42,
    )

    patients = PatientProfileGenerator().generate(config)

    pending_patients = [
        patient
        for patient in patients
        if classify_patient(patient).category == TriageCategory.PENDING
    ]

    assert len(pending_patients) == 4

    for patient in pending_patients:
        assert patient.respiratory_rate is None


def test_red_profiles_support_controlled_variation():
    generator = PatientProfileGenerator()
    red_profiles = defaultdict(set)

    for seed in range(100):
        config = ScenarioConfig(
            scenario_id=f"MCI-RED-{seed}",
            seed=seed,
        )

        patients = generator.generate(config)

        for patient in patients:
            if classify_patient(patient).category == TriageCategory.RED:
                red_profiles["pulse"].add(patient.radial_pulse_present)
                red_profiles["mental_status"].add(patient.mental_status)
                red_profiles["hemorrhage"].add(patient.major_hemorrhage)
                red_profiles["refill"].add(patient.capillary_refill_seconds)

    assert len(red_profiles["pulse"]) == 2
    assert len(red_profiles["mental_status"]) > 1
    assert len(red_profiles["hemorrhage"]) == 2
    assert len(red_profiles["refill"]) > 1
