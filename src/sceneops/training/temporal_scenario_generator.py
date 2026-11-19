from __future__ import annotations

import random

from sceneops.simulator.deterioration import PatientChanges
from sceneops.simulator.patients import MentalStatus
from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_config import (
    PatientArchetype,
    ScenarioConfig,
    ScenarioEventSpec,
)
from sceneops.simulator.scenario_factory import create_hybrid_scenario


class TemporalScenarioGenerator:
    """Generate reproducible scenarios for temporal narration training."""

    @staticmethod
    def _create_config(seed: int, *, diverse: bool = False) -> ScenarioConfig:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        if seed < 0:
            raise ValueError("seed must be >= 0.")

        if diverse:
            rng = random.Random(seed)
            green_rate = rng.choice((26, 28, 30, 32))
            yellow_rate = rng.choice((26, 28, 30, 32))
            red_rate = rng.randint(41, 50)
            second_yellow_rate = rng.choice((26, 28, 30, 32))

            event_specs = (
                ScenarioEventSpec(
                    event_id="TEMP-EV001",
                    target_archetype=PatientArchetype.GREEN,
                    earliest_seconds=60,
                    latest_seconds=120,
                    changes=PatientChanges(
                        ambulatory=False,
                        respiratory_rate=green_rate,
                    ),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV002",
                    target_archetype=PatientArchetype.YELLOW,
                    earliest_seconds=150,
                    latest_seconds=210,
                    changes=PatientChanges(
                        radial_pulse_present=False,
                        respiratory_rate=yellow_rate,
                    ),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV003",
                    target_archetype=PatientArchetype.RED,
                    earliest_seconds=240,
                    latest_seconds=300,
                    changes=PatientChanges(respiratory_rate=red_rate),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV004",
                    target_archetype=PatientArchetype.PENDING,
                    earliest_seconds=330,
                    latest_seconds=390,
                    changes=PatientChanges(
                        follows_commands=False,
                        mental_status=MentalStatus.UNRESPONSIVE,
                    ),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV005",
                    target_archetype=PatientArchetype.YELLOW,
                    earliest_seconds=420,
                    latest_seconds=480,
                    changes=PatientChanges(
                        follows_commands=False,
                        mental_status=MentalStatus.UNRESPONSIVE,
                        respiratory_rate=second_yellow_rate,
                    ),
                ),
            )
        else:
            # Retained legacy event profile for V3 compatibility tests.
            event_specs = (
                ScenarioEventSpec(
                    event_id="TEMP-EV001",
                    target_archetype=PatientArchetype.GREEN,
                    earliest_seconds=60,
                    latest_seconds=120,
                    changes=PatientChanges(ambulatory=False),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV002",
                    target_archetype=PatientArchetype.YELLOW,
                    earliest_seconds=150,
                    latest_seconds=210,
                    changes=PatientChanges(radial_pulse_present=False),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV003",
                    target_archetype=PatientArchetype.RED,
                    earliest_seconds=240,
                    latest_seconds=300,
                    changes=PatientChanges(respiratory_rate=45),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV004",
                    target_archetype=PatientArchetype.PENDING,
                    earliest_seconds=330,
                    latest_seconds=390,
                    changes=PatientChanges(
                        mental_status=MentalStatus.UNRESPONSIVE,
                    ),
                ),
                ScenarioEventSpec(
                    event_id="TEMP-EV005",
                    target_archetype=PatientArchetype.YELLOW,
                    earliest_seconds=420,
                    latest_seconds=480,
                    changes=PatientChanges(
                        follows_commands=False,
                        mental_status=MentalStatus.UNRESPONSIVE,
                    ),
                ),
            )

        return ScenarioConfig(
            scenario_id=f"MCI-TEMP-{seed:05d}",
            seed=seed,
            duration_seconds=600,
            event_specs=event_specs,
        )

    def generate(self, seed: int) -> ScenarioDefinition:
        """Generate the stable legacy temporal scenario profile."""
        return create_hybrid_scenario(self._create_config(seed))

    def generate_diverse(self, seed: int) -> ScenarioDefinition:
        """Generate a seeded scenario with varied observable event outcomes."""
        return create_hybrid_scenario(self._create_config(seed, diverse=True))
