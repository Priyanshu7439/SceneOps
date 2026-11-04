
from enum import Enum

from pydantic import BaseModel, ConfigDict

from sceneops.simulator.patients import Patient


class TriageCategory(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLACK = "black"
    PENDING = "pending_assessment"


class TriageResult(BaseModel):
    """Deterministic triage decision and its explanation."""

    model_config = ConfigDict(frozen=True)

    patient_id: str
    category: TriageCategory
    reason: str
    criteria: tuple[str, ...]


def classify_patient(patient: Patient) -> TriageResult:
    """
    Educational START-style triage decision.

    This is a simplified simulation implementation,
    not a clinically validated triage tool.
    """

    # Step 1: Walking wounded
    if patient.ambulatory:
        return TriageResult(
            patient_id=patient.patient_id,
            category=TriageCategory.GREEN,
            reason="Patient is ambulatory.",
            criteria=("ambulatory",),
        )

    # Step 2: Respiratory assessment
    respiratory_rate = patient.respiratory_rate

    if respiratory_rate is None:
        return TriageResult(
            patient_id=patient.patient_id,
            category=TriageCategory.PENDING,
            reason="Respiratory assessment is missing.",
            criteria=("respiratory_assessment_missing",),
        )

    if respiratory_rate == 0:
        if not patient.airway_opened:
            return TriageResult(
                patient_id=patient.patient_id,
                category=TriageCategory.PENDING,
                reason="Airway repositioning assessment is required.",
                criteria=("airway_assessment_required",),
            )

        return TriageResult(
            patient_id=patient.patient_id,
            category=TriageCategory.BLACK,
            reason="Patient remains apneic after airway repositioning.",
            criteria=("apneic_after_airway_repositioning",),
        )

    # Step 3: Respiratory rate
    if respiratory_rate > 30:
        return TriageResult(
            patient_id=patient.patient_id,
            category=TriageCategory.RED,
            reason="Respiratory rate exceeds 30 breaths per minute.",
            criteria=("respiratory_rate_over_30",),
        )

    # Step 4: Perfusion
    poor_perfusion = (
        not patient.radial_pulse_present
        or (
            patient.capillary_refill_seconds is not None
            and patient.capillary_refill_seconds > 2
        )
    )

    if poor_perfusion:
        return TriageResult(
            patient_id=patient.patient_id,
            category=TriageCategory.RED,
            reason="Perfusion assessment indicates poor perfusion.",
            criteria=("poor_perfusion",),
        )

    # Step 5: Mental status
    if not patient.follows_commands:
        return TriageResult(
            patient_id=patient.patient_id,
            category=TriageCategory.RED,
            reason="Patient cannot follow simple commands.",
            criteria=("cannot_follow_commands",),
        )

    # Patient is not walking but passes the remaining checks
    return TriageResult(
        patient_id=patient.patient_id,
        category=TriageCategory.YELLOW,
        reason="Patient passes respiratory, perfusion and mental checks.",
        criteria=("delayed_assessment",),
    )
