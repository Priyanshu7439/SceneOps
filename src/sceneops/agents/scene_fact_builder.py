import re
from decimal import Decimal, InvalidOperation
from typing import ClassVar

from sceneops.agents.contracts import SceneFact
from sceneops.simulator.observations import (
    LearnerObservation,
    ObservationType,
)


class SceneFactBuilder:
    """Convert validated learner observations into approved scene facts."""

    ALLOWED_OBSERVATIONS = frozenset(
        {
            ObservationType.RESPIRATORY_RATE,
            ObservationType.RADIAL_PULSE,
            ObservationType.CAPILLARY_REFILL,
            ObservationType.MENTAL_STATUS,
            ObservationType.AMBULATORY_STATUS,
            ObservationType.VISIBLE_HEMORRHAGE,
            ObservationType.FOLLOWS_COMMANDS,
        }
    )

    ALLOWED_VALUES: ClassVar[dict[ObservationType, frozenset[str]]] = {
        ObservationType.RADIAL_PULSE: frozenset(
            {
                "present",
                "absent",
            }
        ),
        ObservationType.MENTAL_STATUS: frozenset(
            {
                "alert",
                "confused",
                "unresponsive",
            }
        ),
        ObservationType.AMBULATORY_STATUS: frozenset(
            {
                "ambulatory",
                "non_ambulatory",
            }
        ),
        ObservationType.VISIBLE_HEMORRHAGE: frozenset(
            {
                "visible major hemorrhage",
                "no visible major hemorrhage",
            }
        ),
        ObservationType.FOLLOWS_COMMANDS: frozenset(
            {
                "follows commands",
                "does not follow commands",
            }
        ),
    }

    @staticmethod
    def _validate_numeric_value(
        value: str,
        *,
        minimum: int,
        maximum: int,
    ) -> None:
        try:
            number = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("Observation contains an invalid numeric value.") from exc

        if not number.is_finite() or not minimum <= number <= maximum:
            raise ValueError("Observation numeric value is outside allowed bounds.")

    @classmethod
    def _validate_observation(
        cls,
        observation: LearnerObservation,
    ) -> None:
        if not isinstance(observation, LearnerObservation):
            raise TypeError("Expected a LearnerObservation instance.")

        if not isinstance(observation.observation_type, ObservationType):
            raise TypeError("observation_type must be an ObservationType.")

        if not isinstance(observation.patient_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}",
            observation.patient_id,
        ):
            raise ValueError("Invalid patient ID.")

        if (
            isinstance(observation.observed_at_seconds, bool)
            or not isinstance(observation.observed_at_seconds, int)
            or observation.observed_at_seconds < 0
        ):
            raise ValueError("Invalid observation timestamp.")

        if (
            not isinstance(observation.value, str)
            or not observation.value
            or observation.value != observation.value.strip()
        ):
            raise ValueError("Invalid observation value.")

        observation_type = observation.observation_type
        value = observation.value

        if observation_type == ObservationType.RESPIRATORY_RATE:
            cls._validate_numeric_value(
                value,
                minimum=0,
                maximum=80,
            )

        elif observation_type == ObservationType.CAPILLARY_REFILL:
            cls._validate_numeric_value(
                value,
                minimum=0,
                maximum=20,
            )

        elif (
            observation_type in cls.ALLOWED_VALUES
            and value not in cls.ALLOWED_VALUES[observation_type]
        ):
            raise ValueError(f"Invalid value for {observation_type.value}.")

    @classmethod
    def build(
        cls,
        observations: list[LearnerObservation],
    ) -> tuple[SceneFact, ...]:
        if not isinstance(observations, list):
            raise TypeError("Observations must be provided as a list.")

        facts: list[SceneFact] = []

        for observation in observations:
            cls._validate_observation(observation)

            if observation.observation_type not in cls.ALLOWED_OBSERVATIONS:
                continue

            facts.append(
                SceneFact(
                    fact=(
                        f"patient_observation:"
                        f"{observation.patient_id}:"
                        f"{observation.observation_type.value}"
                    ),
                    value=(
                        f"{observation.value} "
                        f"(observed at simulation second "
                        f"{observation.observed_at_seconds})"
                    ),
                )
            )

        return tuple(facts)
