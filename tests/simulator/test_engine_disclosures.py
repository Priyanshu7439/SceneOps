import pytest

from sceneops.agents.disclosure_rules import PatientQuestion
from sceneops.simulator.engine import SimulationEngine
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance


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


def make_engine(*patients: Patient) -> SimulationEngine:
    engine = SimulationEngine(ambulances=[Ambulance(ambulance_id="A001")])
    for patient in patients or (make_patient(),):
        engine.add_patient(patient)
    return engine


def test_engine_builds_packet_without_undisclosed_pain():
    engine = make_engine()

    packet = engine.build_patient_disclosure_packet("P001")

    assert packet.patient_id == "P001"
    assert packet.current_disclosures == []


def test_engine_records_explicit_patient_question():
    engine = make_engine()

    record = engine.record_patient_question("P001", PatientQuestion.PAIN_LEVEL)

    assert record.patient_id == "P001"
    assert record.disclosure.fact == "reported_pain_level"
    assert record.disclosure.value == "7/10"


def test_engine_packet_contains_disclosed_pain():
    engine = make_engine()
    engine.record_patient_question("P001", PatientQuestion.PAIN_LEVEL)

    packet = engine.build_patient_disclosure_packet("P001")

    assert [(fact.fact, fact.value) for fact in packet.current_disclosures] == [
        ("reported_pain_level", "7/10")
    ]


def test_engine_disclosures_are_isolated_between_patients():
    engine = make_engine(make_patient("P001"), make_patient("P002"))
    engine.record_patient_question("P001", PatientQuestion.PAIN_LEVEL)

    first_packet = engine.build_patient_disclosure_packet("P001")
    second_packet = engine.build_patient_disclosure_packet("P002")

    assert len(first_packet.current_disclosures) == 1
    assert second_packet.current_disclosures == []


def test_engine_rejects_unknown_patient_question():
    engine = make_engine()

    with pytest.raises(KeyError, match="Unknown patient"):
        engine.record_patient_question("UNKNOWN", PatientQuestion.PAIN_LEVEL)


def test_engine_rejects_invalid_question_type():
    engine = make_engine()

    with pytest.raises(TypeError, match="PatientQuestion"):
        engine.record_patient_question("P001", "pain_level")


def test_engine_records_disclosure_event():
    engine = make_engine()

    engine.record_patient_question("P001", PatientQuestion.PAIN_LEVEL)

    assert engine.events[-1].event_type == "patient_information_disclosed"
    assert engine.events[-1].patient_id == "P001"


def test_engine_rejects_disclosure_packet_after_completion():
    engine = make_engine()
    engine.start()
    engine.complete()

    with pytest.raises(RuntimeError, match="after completion"):
        engine.build_patient_disclosure_packet("P001")
