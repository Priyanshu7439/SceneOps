from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
)
from sceneops.agents.conversation_controller import (
    PatientConversationController,
)
from sceneops.agents.disclosure_rules import (
    PatientDisclosureRecord,
    PatientQuestion,
)
from sceneops.agents.orchestrator import AgentOrchestrator
from sceneops.agents.scene_fact_builder import SceneFactBuilder
from sceneops.simulator.clock import ClockSnapshot
from sceneops.simulator.engine import (
    SimulationEngine,
    SimulationStatus,
)
from sceneops.simulator.observations import (
    LearnerObservation,
    ObservationType,
)
from sceneops.simulator.triage import TriageResult


class SceneOpsApplicationService:
    """Orchestrates live simulation and learner-facing interactions."""

    def __init__(
        self,
        *,
        engine: SimulationEngine,
        conversation_controller: PatientConversationController,
        agent_orchestrator: AgentOrchestrator | None = None,
    ) -> None:
        self.engine = engine
        self.conversation_controller = conversation_controller
        self.agent_orchestrator = agent_orchestrator

    def start(self) -> None:
        self.engine.start()

    def pause(self) -> None:
        self.engine.pause()

    def resume(self) -> None:
        self.engine.resume()

    def complete(self) -> None:
        self.engine.complete()

        for patient_id in self.engine.patients:
            self.conversation_controller.conversation_manager.clear_patient(patient_id)

    def assess_patient(self, patient_id: str) -> TriageResult:
        return self.engine.assess_patient(patient_id)

    def observe_patient(
        self,
        patient_id: str,
        observation_type: ObservationType,
    ) -> LearnerObservation:
        """Record a learner observation from the simulator's current state."""
        self._require_running()

        return self.engine.observe_patient(
            patient_id,
            observation_type,
        )

    async def narrate_scene(self, user_message: str) -> AgentResponse:
        """Generate narration exclusively from validated learner observations."""
        self._require_running()

        if self.agent_orchestrator is None:
            raise RuntimeError("Scene narration is not configured for this simulation.")

        observations = self.engine.observation_registry.get_all()

        latest_observations: dict[
            tuple[str, ObservationType],
            LearnerObservation,
        ] = {}

        for observation in observations:
            key = (observation.patient_id, observation.observation_type)
            previous = latest_observations.get(key)

            if (
                previous is None
                or observation.observed_at_seconds >= previous.observed_at_seconds
            ):
                latest_observations[key] = observation

        facts = SceneFactBuilder.build(list(latest_observations.values()))

        if not facts:
            raise ValueError(
                "No learner observations are available for scene narration."
            )

        request = AgentRequest(
            task=AgentTask.SCENE_NARRATION,
            user_message=user_message,
            scene_facts=facts,
        )

        return await self.agent_orchestrator.handle(request)

    def record_patient_question(
        self,
        patient_id: str,
        question: PatientQuestion,
    ) -> PatientDisclosureRecord:
        self._require_running()

        return self.engine.record_patient_question(
            patient_id,
            question,
        )

    async def ask_patient(
        self,
        patient_id: str,
        message: str,
        *,
        disclose_age: bool = False,
    ) -> AgentResponse:
        self._require_running()

        patient = self.engine.patients.get(patient_id)
        if patient is None:
            raise KeyError(f"Unknown patient: {patient_id}")

        return await self.conversation_controller.handle(
            patient=patient,
            user_message=message,
            disclosure_registry=self.engine.disclosure_registry,
            disclose_age=disclose_age,
        )

    def advance_time(self, seconds: int = 60) -> ClockSnapshot:
        return self.engine.advance_time(seconds)

    def get_snapshot(self) -> dict:
        return self.engine.get_snapshot()

    def _require_running(self) -> None:
        if self.engine.status != SimulationStatus.RUNNING:
            raise RuntimeError("This operation requires a running simulation.")
