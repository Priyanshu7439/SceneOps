
from dataclasses import dataclass

from sceneops.simulator.engine import SimulationEngine, SimulationStatus
from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.triage import TriageCategory


@dataclass(frozen=True)
class SimulationRunSummary:
    scenario_id: str
    elapsed_seconds: int
    tick_count: int
    final_status: SimulationStatus
    triage_counts: dict[str, int]
    allocation_count: int
    event_count: int
    transported_patient_ids: tuple[str, ...]


class SimulationRunner:
    """Executes a scenario using the deterministic simulation engine."""

    def __init__(
        self,
        scenario: ScenarioDefinition,
        tick_seconds: int = 60,
    ) -> None:
        if tick_seconds <= 0:
            raise ValueError("tick_seconds must be greater than zero.")

        self.scenario = scenario
        self.tick_seconds = tick_seconds
        self.engine: SimulationEngine = ScenarioBuilder.build(scenario)

    def run(self) -> SimulationRunSummary:
        if self.engine.status != SimulationStatus.CREATED:
            raise RuntimeError("This simulation has already been started.")

        self.engine.start()

        # Perform initial assessment for every patient.
        for patient in self.scenario.patients:
            self.engine.assess_patient(patient.patient_id)

        # Allocate all currently available ambulances.
        self._allocate_available_ambulances()

        elapsed_seconds = 0
        tick_count = 0

        while elapsed_seconds < self.scenario.duration_seconds:
            step = min(
                self.tick_seconds,
                self.scenario.duration_seconds - elapsed_seconds,
            )

            self.engine.advance_time(step)

            elapsed_seconds += step
            tick_count += 1

            # Reassessment is handled by the engine after deterioration.
            # Attempt allocation for any newly eligible waiting patients.
            self._allocate_available_ambulances()

        self.engine.complete()

        triage_counts = {
            category.value: sum(
                1
                for result in self.engine.triage_results.values()
                if result.category == category
            )
            for category in TriageCategory
        }

        return SimulationRunSummary(
            scenario_id=self.scenario.scenario_id,
            elapsed_seconds=elapsed_seconds,
            tick_count=tick_count,
            final_status=self.engine.status,
            triage_counts=triage_counts,
            allocation_count=len(self.engine.allocations)
                + len(self.engine.transported_patients),
            event_count=len(self.engine.events),
            transported_patient_ids=tuple(
                sorted(self.engine.transported_patients)
            ),
        )

    def _allocate_available_ambulances(self) -> None:
        while True:
            allocation = self.engine.allocate_next_patient()

            if allocation is None:
                break
