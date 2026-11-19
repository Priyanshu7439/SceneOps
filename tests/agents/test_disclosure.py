import pytest

from sceneops.agents.disclosure import DisclosureFact
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

    assert [fact.fact for fact in packet.current_disclosures] == ["reported_pain_level"]
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


from sceneops.agents.disclosure import ConversationTurn


def test_packet_contains_supplied_conversation_history():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    history = [
        ConversationTurn(
            user_message="How are you feeling?",
            patient_response="I feel scared.",
        ),
        ConversationTurn(
            user_message="Are you in pain?",
            patient_response="Yes, my leg hurts.",
        ),
    ]

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
        conversation_memory=history,
    )

    assert packet.conversation_memory == history
    assert len(packet.conversation_memory) == 2


def test_packet_defaults_to_empty_conversation_history():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
    )

    assert packet.conversation_memory == []


def test_conversation_history_is_separate_from_trusted_disclosures():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    history = [
        ConversationTurn(
            user_message="What is your respiratory rate?",
            patient_response="I don't know.",
        )
    ]

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
        conversation_memory=history,
    )

    assert packet.conversation_memory == history
    assert packet.current_disclosures == []


def test_packet_uses_latest_disclosure_but_registry_preserves_history():
    patient = make_patient()
    registry = PatientDisclosureRegistry()

    registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    patient.update_pain(9)
    registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    packet = DisclosurePacketBuilder.build(
        patient,
        disclosure_registry=registry,
    )

    records = registry.get_for_patient("P001")

    assert len(records) == 2
    assert [record.disclosure.value for record in records] == ["7/10", "9/10"]

    assert len(packet.current_disclosures) == 1
    assert packet.current_disclosures[0].value == "9/10"


def test_disclosure_policy_rejects_unauthorized_facts():
    from sceneops.agents.disclosure_rules import DisclosurePolicy

    unauthorized_fact = DisclosureFact(
        fact="respiratory_rate",
        value="37",
    )

    with pytest.raises(
        ValueError,
        match="Disclosure policy rejected unauthorized facts",
    ):
        DisclosurePolicy.validate([unauthorized_fact])
