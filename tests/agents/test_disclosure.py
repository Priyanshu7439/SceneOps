import pytest

from sceneops.agents.disclosure_rules import (
    DisclosurePacketBuilder,
    PatientDisclosureRegistry,
    PatientQuestion,
)
from sceneops.simulator.patients import InjuryType, Patient


def make_patient(patient_id: str = "P001") -> Patient:
    return Patient(
        patient_id=patient_id,
        age=32,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=24,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
        personality="calm",
        speech_style="plain",
        pain_level=7,
    )


def test_patient_pain_is_not_disclosed_before_question():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
    )

    assert packet.current_disclosures == []


def test_explicit_pain_question_records_patient_answer():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    record = registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    assert record.patient_id == "P001"
    assert record.disclosure.fact == "reported_pain_level"
    assert record.disclosure.value == "7/10"


def test_packet_contains_only_registry_backed_disclosures():
    patient = make_patient()
    registry = PatientDisclosureRegistry()
    registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
    )

    assert [fact.fact for fact in packet.current_disclosures] == [
        "reported_pain_level"
    ]
    assert packet.current_disclosures[0].value == "7/10"


def test_patient_age_is_hidden_by_default():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
    )

    assert packet.profile.age is None


def test_patient_age_can_be_explicitly_disclosed():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
        disclose_age=True,
    )

    assert packet.profile.age == 32


def test_disclosures_are_separated_by_patient():
    first_patient = make_patient("P001")
    second_patient = make_patient("P002")
    registry = PatientDisclosureRegistry()

    registry.record_answer(first_patient, PatientQuestion.PAIN_LEVEL)

    first_packet = DisclosurePacketBuilder.build(
        first_patient,
        disclosure_registry=registry,
    )
    second_packet = DisclosurePacketBuilder.build(
        second_patient,
        disclosure_registry=registry,
    )

    assert len(first_packet.current_disclosures) == 1
    assert second_packet.current_disclosures == []


def test_registry_returns_a_copy_of_patient_records():
    patient = make_patient()
    registry = PatientDisclosureRegistry()
    registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    records = registry.get_for_patient("P001")
    records.clear()

    assert len(registry.get_for_patient("P001")) == 1


def test_builder_rejects_non_registry_input():
    patient = make_patient()

    with pytest.raises(TypeError, match="PatientDisclosureRegistry"):
        DisclosurePacketBuilder.build(
            patient,
            disclosure_registry=[],
        )
