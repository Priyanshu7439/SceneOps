from __future__ import annotations

from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.observations import (
    LearnerObservation,
    ObservationType,
)
from sceneops.simulator.patients import MentalStatus
from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.scenario_factory import (
    DEFAULT_SCENARIO_INDICES,
    create_mass_casualty_scenario,
)
from sceneops.training.temporal_scene_narration_dataset import (
    TemporalSceneNarrationDatasetGenerator,
    TemporalSceneNarrationExample,
)

TEMPORAL_SCENE_GENERATOR_V2_VERSION = "2.0.0"

# Four distinct, directly observable temporal patterns.
TEMPORAL_V2_TARGETS = (
    ("P005", ObservationType.RESPIRATORY_RATE),
    ("P006", ObservationType.FOLLOWS_COMMANDS),
    ("P007", ObservationType.RADIAL_PULSE),
    ("P008", ObservationType.MENTAL_STATUS),
)


class TemporalSceneNarrationDatasetV2Generator:
    """Generate expanded temporal examples from learner observations."""

    def generate(
        self,
        scenario_indices: tuple[int, ...] = DEFAULT_SCENARIO_INDICES,
    ) -> list[TemporalSceneNarrationExample]:

        if not scenario_indices:
            raise ValueError("At least one scenario index is required.")

        if len(scenario_indices) != len(set(scenario_indices)):
            raise ValueError("Scenario indices must be unique.")

        examples: list[TemporalSceneNarrationExample] = []

        for scenario_index in scenario_indices:
            scenario = create_mass_casualty_scenario(scenario_index)
            scenario = self._add_temporal_events(scenario)

            engine = ScenarioBuilder.build(scenario)
            engine.start()

            initial_observations: dict[
                tuple[str, ObservationType],
                LearnerObservation,
            ] = {}

            for patient_id, observation_type in TEMPORAL_V2_TARGETS:
                initial_observations[(patient_id, observation_type)] = (
                    engine.observe_patient(
                        patient_id,
                        observation_type,
                    )
                )

            target_events = self._get_target_events(scenario.deterioration_events)

            current_time = engine.clock.elapsed_seconds

            for event in target_events:
                engine.advance_time(
                    seconds=event.at_seconds - current_time,
                )
                current_time = engine.clock.elapsed_seconds

                target = next(
                    (
                        item
                        for item in TEMPORAL_V2_TARGETS
                        if item[0] == event.patient_id
                    ),
                    None,
                )

                if target is None:
                    raise ValueError(
                        f"Unexpected temporal event for {event.patient_id}."
                    )

                patient_id, observation_type = target

                follow_up_observation = engine.observe_patient(
                    patient_id,
                    observation_type,
                )

                example = TemporalSceneNarrationDatasetGenerator._build_example(
                    scenario_id=scenario.scenario_id,
                    patient_id=patient_id,
                    observation_type=observation_type,
                    initial_observation=initial_observations[target],
                    follow_up_observation=follow_up_observation,
                )

                examples.append(
                    example.model_copy(
                        update={
                            "generator_version": (TEMPORAL_SCENE_GENERATOR_V2_VERSION),
                        }
                    )
                )

        return examples

    @staticmethod
    def _add_temporal_events(
        scenario: ScenarioDefinition,
    ) -> ScenarioDefinition:
        """Add two additional observable changes to the scenario."""

        event_offset = scenario.deterioration_events[0].at_seconds - 120

        additional_events = [
            DeteriorationEvent(
                event_id="TEMP-EV003",
                patient_id="P007",
                at_seconds=360 + event_offset,
                changes=PatientChanges(
                    radial_pulse_present=False,
                ),
            ),
            DeteriorationEvent(
                event_id="TEMP-EV004",
                patient_id="P008",
                at_seconds=480 + event_offset,
                changes=PatientChanges(
                    mental_status=MentalStatus.CONFUSED,
                ),
            ),
        ]

        scenario_data = scenario.model_dump(mode="python")
        scenario_data["deterioration_events"] = [
            *scenario.deterioration_events,
            *additional_events,
        ]

        return ScenarioDefinition.model_validate(scenario_data)

    @staticmethod
    def _get_target_events(
        events: list[DeteriorationEvent],
    ) -> list[DeteriorationEvent]:
        """Find exactly one temporal event for each target patient."""

        target_patient_ids = {patient_id for patient_id, _ in TEMPORAL_V2_TARGETS}

        matching_events = [
            event for event in events if event.patient_id in target_patient_ids
        ]

        for patient_id in target_patient_ids:
            patient_events = [
                event for event in matching_events if event.patient_id == patient_id
            ]

            if len(patient_events) != 1:
                raise ValueError(
                    f"Expected exactly one temporal event for {patient_id}; "
                    f"found {len(patient_events)}."
                )

        if len(matching_events) != len(TEMPORAL_V2_TARGETS):
            raise ValueError("Unexpected number of temporal target events.")

        return sorted(
            matching_events,
            key=lambda event: event.at_seconds,
        )
