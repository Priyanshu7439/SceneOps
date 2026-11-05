import asyncio

import pytest
from pydantic import ValidationError

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    SafetyReview,
    SceneFact,
)
from sceneops.agents.disclosure import (
    DisclosureFact,
    DisclosurePacket,
    PatientProfile,
)
from sceneops.agents.orchestrator import AgentOrchestrator


class FakePatientAgent:
    async def generate(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            task=AgentTask.PATIENT_DIALOGUE,
            agent_name="fake_patient",
            content="My arm hurts.",
        )


class FakeSceneAgent:
    async def generate(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            task=AgentTask.SCENE_NARRATION,
            agent_name="fake_scene",
            content="A responder approaches the patient.",
        )


class FakeReviewer:
    def __init__(self, approved: bool = True) -> None:
        self.approved = approved

    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview:
        return SafetyReview(approved=self.approved)


def make_patient_request() -> AgentRequest:
    packet = DisclosurePacket(
        patient_id="P001",
        profile=PatientProfile(personality="anxious", speech_style="short"),
        current_disclosures=[
            DisclosureFact(fact="reported_pain_level", value="7/10")
        ],
    )
    return AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you feeling?",
        patient_packet=packet,
    )


def make_scene_request() -> AgentRequest:
    return AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe the current scene.",
        scene_facts=(SceneFact(fact="responder_arrival", value="one responder arrived"),),
    )


def test_patient_dialogue_request_requires_disclosure_packet():
    with pytest.raises(ValidationError, match="requires a disclosure packet"):
        AgentRequest(
            task=AgentTask.PATIENT_DIALOGUE,
            user_message="How are you?",
        )


def test_patient_dialogue_request_rejects_scene_facts():
    packet = make_patient_request().patient_packet
    with pytest.raises(ValidationError, match="cannot receive scene facts"):
        AgentRequest(
            task=AgentTask.PATIENT_DIALOGUE,
            user_message="How are you?",
            patient_packet=packet,
            scene_facts=(SceneFact(fact="noise", value="loud"),),
        )


def test_scene_request_requires_approved_scene_facts():
    with pytest.raises(ValidationError, match="requires approved scene facts"):
        AgentRequest(
            task=AgentTask.SCENE_NARRATION,
            user_message="Describe the scene.",
        )


def test_scene_request_rejects_patient_packet():
    with pytest.raises(ValidationError, match="cannot receive a patient packet"):
        AgentRequest(
            task=AgentTask.SCENE_NARRATION,
            user_message="Describe the scene.",
            patient_packet=make_patient_request().patient_packet,
            scene_facts=(SceneFact(fact="noise", value="loud"),),
        )


def test_orchestrator_routes_patient_dialogue_and_approves_response():
    orchestrator = AgentOrchestrator(
        patient_agent=FakePatientAgent(),
        scene_agent=FakeSceneAgent(),
        safety_reviewer=FakeReviewer(approved=True),
    )

    response = asyncio.run(orchestrator.handle(make_patient_request()))

    assert response.agent_name == "fake_patient"
    assert response.content == "My arm hurts."


def test_orchestrator_routes_scene_narration():
    orchestrator = AgentOrchestrator(
        patient_agent=FakePatientAgent(),
        scene_agent=FakeSceneAgent(),
        safety_reviewer=FakeReviewer(approved=True),
    )

    response = asyncio.run(orchestrator.handle(make_scene_request()))

    assert response.agent_name == "fake_scene"
    assert response.task == AgentTask.SCENE_NARRATION


def test_orchestrator_returns_fallback_when_reviewer_rejects():
    orchestrator = AgentOrchestrator(
        patient_agent=FakePatientAgent(),
        scene_agent=FakeSceneAgent(),
        safety_reviewer=FakeReviewer(approved=False),
        rejection_message="Response blocked.",
    )

    response = asyncio.run(orchestrator.handle(make_patient_request()))

    assert response.agent_name == "safety_reviewer"
    assert response.content == "Response blocked."


def test_orchestrator_rejects_mismatched_agent_task():
    class WrongTaskAgent:
        async def generate(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                task=AgentTask.SCENE_NARRATION,
                agent_name="wrong_agent",
                content="Wrong task output.",
            )

    orchestrator = AgentOrchestrator(
        patient_agent=WrongTaskAgent(),
        scene_agent=FakeSceneAgent(),
        safety_reviewer=FakeReviewer(),
    )

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(orchestrator.handle(make_patient_request()))
