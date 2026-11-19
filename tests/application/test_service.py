import asyncio

import pytest

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    SafetyReview,
)
from sceneops.agents.conversation_controller import (
    PatientConversationController,
)
from sceneops.agents.conversation_memory import PatientConversationManager
from sceneops.agents.disclosure import ConversationTurn
from sceneops.agents.disclosure_rules import (
    PatientQuestion,
)
from sceneops.agents.orchestrator import AgentOrchestrator
from sceneops.application.service import SceneOpsApplicationService
from sceneops.simulator.engine import SimulationEngine, SimulationStatus
from sceneops.simulator.observations import ObservationType
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance


def make_patient(patient_id: str = "P001") -> Patient:
    return Patient(
        patient_id=patient_id,
        age=32,
        injury_type=InjuryType.TRAUMA,
        ambulatory=False,
        respiratory_rate=24,
        radial_pulse_present=True,
        follows_commands=True,
        major_hemorrhage=False,
        personality="anxious",
        speech_style="plain",
        pain_level=7,
    )


class FakePatientAgent:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)

        return AgentResponse(
            task=AgentTask.PATIENT_DIALOGUE,
            agent_name="fake_patient_agent",
            content="I feel scared.",
        )


class FakeSafetyReviewer:
    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview:
        return SafetyReview(approved=True, reasons=())


def make_service(
    *patients: Patient,
) -> tuple[
    SceneOpsApplicationService,
    SimulationEngine,
    PatientConversationManager,
    FakePatientAgent,
]:
    engine = SimulationEngine(ambulances=[Ambulance(ambulance_id="A001")])

    for patient in patients or (make_patient(),):
        engine.add_patient(patient)

    manager = PatientConversationManager()
    patient_agent = FakePatientAgent()

    controller = PatientConversationController(
        patient_agent=patient_agent,
        safety_reviewer=FakeSafetyReviewer(),
        conversation_manager=manager,
    )

    service = SceneOpsApplicationService(
        engine=engine,
        conversation_controller=controller,
    )

    return service, engine, manager, patient_agent


def test_service_starts_simulation():
    service, engine, _, _ = make_service()

    service.start()

    assert engine.status == SimulationStatus.RUNNING


def test_service_pauses_and_resumes_simulation():
    service, engine, _, _ = make_service()

    service.start()
    service.pause()

    assert engine.status == SimulationStatus.PAUSED

    service.resume()

    assert engine.status == SimulationStatus.RUNNING


def test_service_completes_simulation():
    service, engine, _, _ = make_service()

    service.start()
    service.complete()

    assert engine.status == SimulationStatus.COMPLETED


def test_service_rejects_dialogue_before_start():
    service, _, _, _ = make_service()

    with pytest.raises(RuntimeError, match="running simulation"):
        asyncio.run(service.ask_patient("P001", "How are you feeling?"))


def test_service_rejects_dialogue_while_paused():
    service, _, _, _ = make_service()

    service.start()
    service.pause()

    with pytest.raises(RuntimeError, match="running simulation"):
        asyncio.run(service.ask_patient("P001", "How are you feeling?"))


def test_service_rejects_unknown_patient():
    service, _, _, _ = make_service()

    service.start()

    with pytest.raises(KeyError, match="Unknown patient"):
        asyncio.run(service.ask_patient("UNKNOWN", "How are you feeling?"))


def test_service_returns_patient_dialogue():
    service, _, manager, patient_agent = make_service()

    service.start()

    response = asyncio.run(service.ask_patient("P001", "How are you feeling?"))

    assert response.content == "I feel scared."
    assert response.agent_name == "fake_patient_agent"
    assert len(manager.get_history("P001")) == 1
    assert len(patient_agent.requests) == 1


def test_service_records_supported_patient_question():
    service, engine, _, _ = make_service()

    service.start()

    record = service.record_patient_question(
        "P001",
        PatientQuestion.PAIN_LEVEL,
    )

    assert record.disclosure.fact == "reported_pain_level"
    assert record.disclosure.value == "7/10"
    assert len(engine.disclosure_registry.get_for_patient("P001")) == 1


def test_service_advances_simulation_time():
    service, engine, _, _ = make_service()

    service.start()

    snapshot = service.advance_time(120)

    assert snapshot.elapsed_seconds == 120
    assert snapshot.tick_count == 1
    assert engine.clock.elapsed_seconds == 120


def test_service_assesses_patient_before_start():
    service, engine, _, _ = make_service()

    result = service.assess_patient("P001")

    assert result.patient_id == "P001"
    assert "P001" in engine.triage_results


def test_service_rejects_time_advance_while_paused():
    service, _, _, _ = make_service()

    service.start()
    service.pause()

    with pytest.raises(RuntimeError, match="running"):
        service.advance_time(60)


def test_service_snapshot_contains_engine_state():
    service, _, _, _ = make_service()

    snapshot = service.get_snapshot()

    assert snapshot["status"] == "created"
    assert snapshot["patient_count"] == 1
    assert snapshot["clock"]["elapsed_seconds"] == 0


def test_service_clears_only_its_patient_histories_on_completion():
    service, engine, manager, _ = make_service(
        make_patient("P001"),
        make_patient("P002"),
    )

    manager.add_turn(
        "P001",
        ConversationTurn(
            user_message="How are you?",
            patient_response="I feel scared.",
        ),
    )
    manager.add_turn(
        "P002",
        ConversationTurn(
            user_message="Can you walk?",
            patient_response="No, I cannot.",
        ),
    )
    manager.add_turn(
        "OTHER_SIMULATION_PATIENT",
        ConversationTurn(
            user_message="Are you okay?",
            patient_response="I am okay.",
        ),
    )

    service.start()
    service.complete()

    assert engine.status == SimulationStatus.COMPLETED
    assert manager.get_history("P001") == []
    assert manager.get_history("P002") == []
    assert len(manager.get_history("OTHER_SIMULATION_PATIENT")) == 1


def test_service_rejects_patient_question_while_paused():
    service, _, _, _ = make_service()

    service.start()
    service.pause()

    with pytest.raises(RuntimeError, match="running simulation"):
        service.record_patient_question(
            "P001",
            PatientQuestion.PAIN_LEVEL,
        )


def test_service_records_patient_observation():
    service, engine, _, _ = make_service()

    service.start()

    observation = service.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    assert observation.patient_id == "P001"
    assert observation.observation_type == ObservationType.RESPIRATORY_RATE
    assert observation.value == "24"
    assert observation.observed_at_seconds == 0

    recorded = engine.observation_registry.get_for_patient("P001")
    assert len(recorded) == 1
    assert recorded[0] == observation


def test_service_rejects_observation_before_start():
    service, _, _, _ = make_service()

    with pytest.raises(RuntimeError, match="running simulation"):
        service.observe_patient(
            "P001",
            ObservationType.RESPIRATORY_RATE,
        )


def test_service_rejects_observation_while_paused():
    service, _, _, _ = make_service()

    service.start()
    service.pause()

    with pytest.raises(RuntimeError, match="running simulation"):
        service.observe_patient(
            "P001",
            ObservationType.RESPIRATORY_RATE,
        )


def test_service_rejects_observation_for_unknown_patient():
    service, _, _, _ = make_service()

    service.start()

    with pytest.raises(KeyError, match="Unknown patient"):
        service.observe_patient(
            "UNKNOWN",
            ObservationType.RESPIRATORY_RATE,
        )


def test_service_rejects_invalid_observation_type():
    service, _, _, _ = make_service()

    service.start()

    with pytest.raises(TypeError):
        service.observe_patient(
            "P001",
            "respiratory_rate",
        )


class FakeSceneAgent:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            task=AgentTask.SCENE_NARRATION,
            agent_name="fake_scene_agent",
            content="P001 has a respiratory rate of 24.",
        )


class FakeOrchestratorReviewer:
    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview:
        return SafetyReview(approved=True, reasons=())


def make_narration_service():
    service, engine, _, _ = make_service()

    scene_agent = FakeSceneAgent()
    orchestrator = AgentOrchestrator(
        patient_agent=FakePatientAgent(),
        scene_agent=scene_agent,
        safety_reviewer=FakeOrchestratorReviewer(),
    )

    service.agent_orchestrator = orchestrator
    return service, engine, scene_agent


def test_service_generates_scene_narration_from_observations():
    service, _engine, scene_agent = make_narration_service()
    service.start()

    service.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    response = asyncio.run(service.narrate_scene("Summarize the current scene."))

    assert response.agent_name == "fake_scene_agent"
    assert response.task == AgentTask.SCENE_NARRATION
    assert len(scene_agent.requests) == 1

    fact = scene_agent.requests[0].scene_facts[0]
    assert fact.fact == "patient_observation:P001:respiratory_rate"
    assert "24" in fact.value


def test_service_uses_latest_observation_for_each_type():
    service, engine, scene_agent = make_narration_service()
    service.start()

    service.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    engine.advance_time(60)

    service.observe_patient(
        "P001",
        ObservationType.RESPIRATORY_RATE,
    )

    asyncio.run(service.narrate_scene("Summarize the scene."))

    facts = scene_agent.requests[0].scene_facts

    respiratory_facts = [
        fact for fact in facts if fact.fact.endswith(":respiratory_rate")
    ]

    assert len(respiratory_facts) == 1
    assert "observed at simulation second 60" in respiratory_facts[0].value


def test_service_rejects_narration_without_observations():
    service, _, _ = make_narration_service()
    service.start()

    with pytest.raises(ValueError, match="No learner observations"):
        asyncio.run(service.narrate_scene("Summarize the scene."))


def test_service_rejects_narration_before_start():
    service, _, _ = make_narration_service()

    with pytest.raises(RuntimeError, match="running simulation"):
        asyncio.run(service.narrate_scene("Summarize the scene."))


def test_service_rejects_narration_when_orchestrator_is_missing():
    service, _, _, _ = make_service()
    service.start()

    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(service.narrate_scene("Summarize the scene."))
