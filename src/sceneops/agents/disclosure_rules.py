from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from sceneops.agents.disclosure import (
    ConversationTurn,
    DisclosureFact,
    DisclosurePacket,
    PatientProfile,
)
from sceneops.simulator.patients import Patient


class PatientQuestion(str, Enum):
    """Questions that the current simulator can answer from patient state."""

    PAIN_LEVEL = "pain_level"


class PatientDisclosureRecord(BaseModel):
    """A patient fact recorded after an explicit learner question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str = Field(min_length=1)
    disclosure: DisclosureFact


class DisclosurePolicy:
    """Allowlist of facts permitted in patient dialogue packets."""

    ALLOWED_FACTS = frozenset({"reported_pain_level"})

    @classmethod
    def validate(cls, facts: list[DisclosureFact]) -> list[DisclosureFact]:
        rejected = [fact.fact for fact in facts if fact.fact not in cls.ALLOWED_FACTS]
        if rejected:
            raise ValueError(
                f"Disclosure policy rejected unauthorized facts: {rejected}"
            )
        return list(facts)


class PatientDisclosureRegistry:
    """Stores facts the simulated patient has explicitly disclosed.

    Historical records are retained for audit purposes. The packet builder
    selects the latest record for each fact before sending data to an agent.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[PatientDisclosureRecord]] = {}

    def record_answer(
        self,
        patient: Patient,
        question: PatientQuestion,
    ) -> PatientDisclosureRecord:
        if not isinstance(question, PatientQuestion):
            raise TypeError("question must be a PatientQuestion.")

        match question:
            case PatientQuestion.PAIN_LEVEL:
                fact = DisclosureFact(
                    fact="reported_pain_level",
                    value=f"{patient.pain_level}/10",
                )
            case _:
                raise ValueError(f"Unsupported patient question: {question}")

        record = PatientDisclosureRecord(
            patient_id=patient.patient_id,
            disclosure=fact,
        )
        self._records.setdefault(patient.patient_id, []).append(record)
        return record

    def get_for_patient(self, patient_id: str) -> list[PatientDisclosureRecord]:
        return list(self._records.get(patient_id, []))

    def clear_patient(self, patient_id: str) -> None:
        self._records.pop(patient_id, None)


class DisclosurePacketBuilder:
    """Builds packets using only the latest registry-backed disclosures."""

    @staticmethod
    def _latest_disclosures(
        records: list[PatientDisclosureRecord],
    ) -> list[DisclosureFact]:
        latest_by_fact: dict[str, DisclosureFact] = {}

        for record in records:
            latest_by_fact[record.disclosure.fact] = record.disclosure

        return DisclosurePolicy.validate(list(latest_by_fact.values()))

    @staticmethod
    def build(
        patient: Patient,
        *,
        disclosure_registry: PatientDisclosureRegistry,
        disclose_age: bool = False,
        conversation_memory: list[ConversationTurn] | None = None,
    ) -> DisclosurePacket:
        if not isinstance(disclosure_registry, PatientDisclosureRegistry):
            raise TypeError("disclosure_registry must be a PatientDisclosureRegistry.")

        records = disclosure_registry.get_for_patient(patient.patient_id)
        current_disclosures = DisclosurePacketBuilder._latest_disclosures(records)

        profile = PatientProfile(
            age=patient.age if disclose_age else None,
            personality=patient.personality,
            speech_style=patient.speech_style,
        )

        return DisclosurePacket(
            patient_id=patient.patient_id,
            profile=profile,
            current_disclosures=current_disclosures,
            conversation_memory=conversation_memory or [],
        )
