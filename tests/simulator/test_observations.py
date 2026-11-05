
import pytest
from pydantic import ValidationError

from sceneops.simulator.observations import (
    LearnerObservation,
    LearnerObservationRegistry,
    ObservationType,
)


def make_observation(
    patient_id: str = "P001",
    observation_type: ObservationType = ObservationType.RESPIRATORY_RATE,
    value: str = "37",
    observed_at_seconds: int = 120,
) -> LearnerObservation:
    return LearnerObservation(
        patient_id=patient_id,
        observation_type=observation_type,
        value=value,
        observed_at_seconds=observed_at_seconds,
    )


def test_record_observation():
    registry = LearnerObservationRegistry()
    observation = make_observation()

    registry.record(observation)

    assert registry.get_for_patient("P001") == [observation]


def test_observations_are_separated_by_patient():
    registry = LearnerObservationRegistry()

    first = make_observation(patient_id="P001")
    second = make_observation(patient_id="P002")

    registry.record(first)
    registry.record(second)

    assert registry.get_for_patient("P001") == [first]
    assert registry.get_for_patient("P002") == [second]


def test_multiple_observations_are_preserved_in_order():
    registry = LearnerObservationRegistry()

    first = make_observation(value="37", observed_at_seconds=120)
    second = make_observation(
        observation_type=ObservationType.RADIAL_PULSE,
        value="absent",
        observed_at_seconds=150,
    )

    registry.record(first)
    registry.record(second)

    assert registry.get_for_patient("P001") == [first, second]


def test_get_all_returns_all_observations():
    registry = LearnerObservationRegistry()

    first = make_observation(patient_id="P001")
    second = make_observation(patient_id="P002")

    registry.record(first)
    registry.record(second)

    assert registry.get_all() == [first, second]


def test_unknown_patient_returns_empty_list():
    registry = LearnerObservationRegistry()

    assert registry.get_for_patient("UNKNOWN") == []


def test_clear_patient_removes_only_that_patient():
    registry = LearnerObservationRegistry()

    first = make_observation(patient_id="P001")
    second = make_observation(patient_id="P002")

    registry.record(first)
    registry.record(second)

    registry.clear_patient("P001")

    assert registry.get_for_patient("P001") == []
    assert registry.get_for_patient("P002") == [second]


def test_negative_observation_time_is_rejected():
    with pytest.raises(ValidationError):
        make_observation(observed_at_seconds=-1)


def test_empty_observation_value_is_rejected():
    with pytest.raises(ValidationError):
        make_observation(value="")
