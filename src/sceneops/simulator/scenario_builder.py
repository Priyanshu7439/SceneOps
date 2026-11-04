
from sceneops.simulator.engine import SimulationEngine
from sceneops.simulator.scenario import ScenarioDefinition


class ScenarioBuilder:
    """Builds a simulation engine from a validated scenario definition."""

    @staticmethod
    def build(scenario: ScenarioDefinition) -> SimulationEngine:
        engine = SimulationEngine(
            ambulances=scenario.ambulances,
            deterioration_events=scenario.deterioration_events,
        )

        for patient in scenario.patients:
            engine.add_patient(patient)

        return engine
