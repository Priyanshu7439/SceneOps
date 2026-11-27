
from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    TemporalObservation,
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
from sceneops.simulator.resources import AllocationResult
from sceneops.simulator.transport import TransportRecord
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

    def get_resource_dashboard(self) -> list[dict]:
        return [
            {
                "ambulance_id": ambulance.ambulance_id,
                "status": ambulance.status.value,
                "assigned_patient_id": ambulance.assigned_patient_id,
            }
            for ambulance in self.engine.resource_manager.ambulances.values()
        ]

    def get_event_timeline(self) -> list[dict]:
        return [
            {
                "event_type": event.event_type,
                "simulation_second": event.simulation_second,
                "description": event.description,
                "patient_id": event.patient_id,
            }
            for event in self.engine.events
        ]

    def assess_patient(self, patient_id: str) -> TriageResult:
        return self.engine.assess_patient(patient_id)

    def allocate_patient(self, patient_id: str) -> AllocationResult | None:
        self._require_running()
        return self.engine.allocate_patient(patient_id)

    def start_transport(
        self,
        patient_id: str,
        travel_duration_seconds: int,
    ) -> TransportRecord:
        self._require_running()
        return self.engine.start_transport(
            patient_id,
            travel_duration_seconds,
        )

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

    async def narrate_temporal_scene(
        self,
        user_message: str,
    ) -> AgentResponse:
        """Narrate the complete timestamped learner observation history."""
        self._require_running()

        if self.agent_orchestrator is None:
            raise RuntimeError(
                "Temporal scene narration is not configured for this simulation."
            )

        observations = self.engine.observation_registry.get_all()

        if not observations:
            raise ValueError(
                "No learner observations are available for temporal narration."
            )

        temporal_observations = tuple(
            TemporalObservation(
                patient_id=observation.patient_id,
                observation_type=observation.observation_type.value,
                value=observation.value,
                observed_at_seconds=observation.observed_at_seconds,
            )
            for observation in sorted(
                observations,
                key=lambda item: item.observed_at_seconds,
            )
        )

        request = AgentRequest(
            task=AgentTask.TEMPORAL_SCENE_NARRATION,
            user_message=user_message,
            temporal_observations=temporal_observations,
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

    def get_patient_dashboard(self) -> list[dict]:
        """Return safe, learner-facing patient dashboard metadata."""
        engine = self.engine
        patients = []

        for patient_id in engine.patients:
            triage = engine.triage_results.get(patient_id)
            allocation = engine.allocations.get(patient_id)

            latest_transport = max(
                (
                    record
                    for record in engine.transport_manager.transports.values()
                    if record.patient_id == patient_id
                ),
                key=lambda record: record.started_at_seconds,
                default=None,
            )

            if patient_id in engine.transported_patients:
                transport_status = "transported"
            elif latest_transport is not None:
                transport_status = latest_transport.status.value
            elif allocation is not None:
                transport_status = "allocated"
            else:
                transport_status = "not_allocated"

            ambulance_id = (
                allocation.ambulance_id
                if allocation is not None
                else (
                    latest_transport.ambulance_id
                    if latest_transport is not None
                    else None
                )
            )

            patients.append({
                "patient_id": patient_id,
                "assessment_status": (
                    "assessed" if triage is not None else "pending"
                ),
                "triage_category": (
                    triage.category.value
                    if triage is not None
                    else "pending_assessment"
                ),
                "ambulance_id": ambulance_id,
                "transport_status": transport_status,
            })

        return patients

    def _require_running(self) -> None:
        if self.engine.status != SimulationStatus.RUNNING:
            raise RuntimeError("This operation requires a running simulation.")
