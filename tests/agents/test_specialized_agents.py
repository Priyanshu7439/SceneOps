import asyncio

import pytest

from sceneops.agents.contracts import AgentRequest, AgentTask, SceneFact
from sceneops.agents.disclosure import (
    DisclosureFact,
    DisclosurePacket,
    PatientProfile,
)
from sceneops.agents.patient_dialogue import PatientDialogueAgent
from sceneops.agents.safety_reviewer import RuleBasedSafetyReviewer
from sceneops.agents.scene_narration import SceneNarrationAgent


class StubBackend:
    def __init__(self, response: str = "I feel scared.") -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


def make_packet() -> DisclosurePacket:
    return DisclosurePacket(
        patient_id="P001",
        profile=PatientProfile(age=None, personality="anxious", speech_style="short sentences"),
        current_disclosures=[
            DisclosureFact(fact="reported_pain_level", value="5/10")
        ],
    )


def test_patient_agent_uses_disclosure_packet_not_patient_object():
    backend = StubBackend("It hurts a little. I am scared.")
    agent = PatientDialogueAgent(backend)
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you feeling?",
        patient_packet=make_packet(),
    )

    response = asyncio.run(agent.generate(request))

    assert response.content == "It hurts a little. I am scared."
    assert "5/10" in backend.system_prompt
    assert "age: not disclosed" in backend.system_prompt
    assert response.agent_name == "patient_dialogue_agent"


def test_patient_agent_rejects_wrong_task():
    agent = PatientDialogueAgent(StubBackend())
    request = AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe this scene.",
        scene_facts=(SceneFact(fact="weather", value="rainy"),),
    )

    with pytest.raises(ValueError, match="only accepts patient_dialogue"):
        asyncio.run(agent.generate(request))


def test_patient_agent_rejects_empty_backend_response():
    agent = PatientDialogueAgent(StubBackend("   "))
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you?",
        patient_packet=make_packet(),
    )

    with pytest.raises(ValueError, match="empty response"):
        asyncio.run(agent.generate(request))


def test_scene_agent_uses_only_approved_scene_facts():
    backend = StubBackend("Rain is falling near the staging area.")
    agent = SceneNarrationAgent(backend)
    request = AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe the visible conditions.",
        scene_facts=(SceneFact(fact="weather", value="rain"),),
    )

    response = asyncio.run(agent.generate(request))

    assert "weather: rain" in backend.system_prompt
    assert response.task == AgentTask.SCENE_NARRATION
    assert response.agent_name == "scene_narration_agent"


def test_scene_agent_rejects_wrong_task():
    agent = SceneNarrationAgent(StubBackend())
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you?",
        patient_packet=make_packet(),
    )

    with pytest.raises(ValueError, match="only accepts scene_narration"):
        asyncio.run(agent.generate(request))


def test_safety_reviewer_approves_plain_patient_response():
    reviewer = RuleBasedSafetyReviewer()
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you?",
        patient_packet=make_packet(),
    )
    response = asyncio.run(PatientDialogueAgent(StubBackend()).generate(request))

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is True
    assert review.reasons == ()


def test_safety_reviewer_rejects_diagnosis_or_treatment_claim():
    reviewer = RuleBasedSafetyReviewer()
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you?",
        patient_packet=make_packet(),
    )
    response = asyncio.run(PatientDialogueAgent(
        StubBackend("You have a heart attack. You should take medication.")
    ).generate(request))

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is False
    assert review.reasons


def test_safety_reviewer_rejects_undisclosed_vital_measurement():
    reviewer = RuleBasedSafetyReviewer()
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How is your breathing?",
        patient_packet=make_packet(),
    )
    response = asyncio.run(PatientDialogueAgent(
        StubBackend("My respiratory rate is 35.")
    ).generate(request))

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is False
    assert any("undisclosed clinical measurement" in reason for reason in review.reasons)


def test_safety_reviewer_allows_disclosed_pain_value():
    reviewer = RuleBasedSafetyReviewer()
    request = AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How much pain are you in?",
        patient_packet=make_packet(),
    )
    response = asyncio.run(PatientDialogueAgent(
        StubBackend("My pain is 5/10.")
    ).generate(request))

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is True
