import asyncio
from copy import deepcopy

import pytest

from sceneops.agents.contracts import AgentRequest, AgentResponse, AgentTask
from sceneops.application.service import SceneOpsApplicationService
from sceneops.simulator.engine import SimulationEngine
from sceneops.simulator.observations import ObservationType
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance


class FakeOrchestrator:
    def __init__(self, error: Exception | None = None):
        self.requests: list[AgentRequest] = []
        self.error = error

    async def handle(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if self.error:
            raise self.error
        return AgentResponse(
            task=request.task,
            agent_name="fake",
            content="Approved observations only.",
        )


def make_service(orchestrator: FakeOrchestrator):
    engine = SimulationEngine([Ambulance(ambulance_id="A001")])
    engine.add_patient(
        Patient(
            patient_id="P001",
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
    )
    service = SceneOpsApplicationService(
        engine=engine,
        conversation_controller=object(),
        agent_orchestrator=orchestrator,
    )
    service.start()
    service.observe_patient("P001", ObservationType.RESPIRATORY_RATE)
    return service, engine


def test_narration_receives_disclosed_facts_only_and_cannot_mutate_state():
    orchestrator = FakeOrchestrator()
    service, engine = make_service(orchestrator)
    before = deepcopy(engine.get_snapshot())
    response = asyncio.run(service.narrate_scene("Narrate the scene."))

    assert response.content == "Approved observations only."
    assert len(orchestrator.requests) == 1
    request = orchestrator.requests[0]
    assert request.task == AgentTask.SCENE_NARRATION
    assert len(request.scene_facts) == 1
    assert request.scene_facts[0].fact == (
        "patient_observation:P001:respiratory_rate"
    )
    assert request.scene_facts[0].value.startswith("24")
    assert not hasattr(request, "patient_state")
    assert not hasattr(request, "ambulances")
    assert not hasattr(request, "survival_outcome")
    assert engine.get_snapshot() == before


def test_model_failure_does_not_change_clock_resources_or_patients():
    orchestrator = FakeOrchestrator(TimeoutError("model timeout"))
    service, engine = make_service(orchestrator)
    before = deepcopy(engine.get_snapshot())

    with pytest.raises(TimeoutError, match="model timeout"):
        asyncio.run(service.narrate_scene("Narrate the scene."))

    assert engine.get_snapshot() == before
    assert engine.resource_manager.available_ambulances()
