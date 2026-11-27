import asyncio

import pytest

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    SceneFact,
    TemporalObservation,
)
from sceneops.agents.disclosure import (
    DisclosureFact,
    DisclosurePacket,
    PatientProfile,
)
from sceneops.agents.patient_dialogue import PatientDialogueAgent
from sceneops.agents.safety_reviewer import RuleBasedSafetyReviewer
from sceneops.agents.scene_narration import SceneNarrationAgent
from sceneops.agents.temporal_scene_narration import TemporalSceneNarrationAgent


class StubBackend:
    def __init__(self, response: str = '{"style": "plain"}') -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


def make_packet(
    *,
    pain: str | None = "5/10",
    age: int | None = None,
) -> DisclosurePacket:
    disclosures = []

    if pain is not None:
        disclosures.append(DisclosureFact(fact="reported_pain_level", value=pain))

    return DisclosurePacket(
        patient_id="P001",
        profile=PatientProfile(
            age=age,
            personality="anxious",
            speech_style="short sentences",
        ),
        current_disclosures=disclosures,
    )


def make_patient_request(
    message: str,
    *,
    packet: DisclosurePacket | None = None,
) -> AgentRequest:
    return AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message=message,
        patient_packet=packet or make_packet(),
    )


def test_patient_agent_renders_only_disclosed_pain():
    agent = PatientDialogueAgent(StubBackend())
    request = make_patient_request("How much pain are you in?")

    response = asyncio.run(agent.generate(request))

    assert response.content == "My pain is around 5/10."
    assert response.agent_name == "patient_dialogue_agent"


def test_patient_agent_does_not_invent_undisclosed_symptoms():
    agent = PatientDialogueAgent(StubBackend())
    request = make_patient_request("Do you have a headache?")

    response = asyncio.run(agent.generate(request))

    assert response.content == "I don't know. Nobody has told me that."
    assert "headache" not in response.content.casefold()


def test_patient_agent_returns_unknown_when_pain_is_not_disclosed():
    agent = PatientDialogueAgent(StubBackend())
    request = make_patient_request(
        "How much pain are you in?",
        packet=make_packet(pain=None),
    )

    response = asyncio.run(agent.generate(request))

    assert response.content == "I don't know. Nobody has told me that."


def test_patient_agent_renders_disclosed_age():
    agent = PatientDialogueAgent(StubBackend())
    request = make_patient_request(
        "How old are you?",
        packet=make_packet(age=32),
    )

    response = asyncio.run(agent.generate(request))

    assert response.content == "I'm 32 years old."


def test_patient_agent_uses_selected_dialogue_style():
    agent = PatientDialogueAgent(StubBackend('{"style": "hesitant"}'))
    request = make_patient_request("How much pain are you in?")

    response = asyncio.run(agent.generate(request))

    assert response.content == "Um... I think my pain is around 5/10."


def test_patient_agent_falls_back_on_invalid_model_output():
    agent = PatientDialogueAgent(StubBackend("not valid json"))
    request = make_patient_request("How much pain are you in?")

    response = asyncio.run(agent.generate(request))

    assert response.content == "My pain is around 5/10."


def test_patient_agent_rejects_wrong_task():
    agent = PatientDialogueAgent(StubBackend())
    request = AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe this scene.",
        scene_facts=(SceneFact(fact="weather", value="rainy"),),
    )

    with pytest.raises(ValueError, match="only accepts patient_dialogue"):
        asyncio.run(agent.generate(request))


def test_scene_agent_uses_only_approved_scene_facts():
    class SceneBackend:
        async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            self.system_prompt = system_prompt
            return "Rain is falling near the staging area."

    backend = SceneBackend()
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
    request = make_patient_request("How much pain are you in?")
    response = asyncio.run(PatientDialogueAgent(StubBackend()).generate(request))

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is True
    assert review.reasons == ()


def test_safety_reviewer_rejects_diagnosis_or_treatment_claim():
    reviewer = RuleBasedSafetyReviewer()
    request = make_patient_request("How are you?")

    response = AgentResponse(
        task=AgentTask.PATIENT_DIALOGUE,
        agent_name="test_patient_agent",
        content="You have a heart attack. You should take medication.",
    )

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is False
    assert review.reasons


def test_safety_reviewer_rejects_undisclosed_vital_measurement():
    reviewer = RuleBasedSafetyReviewer()
    request = make_patient_request("How is your breathing?")

    response = AgentResponse(
        task=AgentTask.PATIENT_DIALOGUE,
        agent_name="test_patient_agent",
        content="My respiratory rate is 35.",
    )

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is False
    assert any(
        "undisclosed clinical measurement" in reason for reason in review.reasons
    )


def test_safety_reviewer_allows_disclosed_pain_value():
    reviewer = RuleBasedSafetyReviewer()
    request = make_patient_request("How much pain are you in?")
    response = asyncio.run(PatientDialogueAgent(StubBackend()).generate(request))

    review = asyncio.run(reviewer.review(request, response))

    assert review.approved is True


def test_scene_agent_currently_allows_unsupported_diagnosis():
    agent = SceneNarrationAgent(StubBackend("Patient P001 has internal bleeding."))
    request = AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe the scene.",
        scene_facts=(SceneFact(fact="weather", value="rainy"),),
    )

    response = asyncio.run(agent.generate(request))

    # Documents the current gap: scene output is not independently reviewed.
    assert "internal bleeding" in response.content.casefold()

def make_temporal_request() -> AgentRequest:
    return AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe the recorded changes over time.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
            TemporalObservation(
                patient_id="P001",
                observation_type="respiratory_rate",
                value="24",
                observed_at_seconds=60,
            ),
            TemporalObservation(
                patient_id="P002",
                observation_type="mental_status",
                value="alert",
                observed_at_seconds=60,
            ),
        ),
    )


def test_temporal_agent_includes_all_timestamped_observations():
    backend = StubBackend("Recorded observations show two readings.")
    agent = TemporalSceneNarrationAgent(backend)

    response = asyncio.run(agent.generate(make_temporal_request()))

    assert "Simulation second 0" in backend.system_prompt
    assert "Simulation second 60" in backend.system_prompt
    assert "P001" in backend.system_prompt
    assert "P002" in backend.system_prompt
    assert "respiratory_rate = 20" in backend.system_prompt
    assert "respiratory_rate = 24" in backend.system_prompt
    assert "mental_status = alert" in backend.system_prompt
    assert response.task == AgentTask.TEMPORAL_SCENE_NARRATION
    assert response.agent_name == "temporal_scene_narration_agent"


def test_temporal_agent_preserves_chronological_observation_order():
    backend = StubBackend("Recorded observations show changes.")
    agent = TemporalSceneNarrationAgent(backend)

    asyncio.run(agent.generate(make_temporal_request()))

    first = backend.system_prompt.index("Simulation second 0")
    second = backend.system_prompt.index("Simulation second 60")

    assert first < second


def test_temporal_agent_rejects_wrong_task():
    agent = TemporalSceneNarrationAgent(StubBackend())
    request = AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe the scene.",
        scene_facts=(SceneFact(fact="weather", value="rainy"),),
    )

    with pytest.raises(
        ValueError,
        match="only accepts temporal_scene_narration",
    ):
        asyncio.run(agent.generate(request))


def test_temporal_agent_rejects_empty_backend_response():
    agent = TemporalSceneNarrationAgent(StubBackend("   "))

    with pytest.raises(ValueError, match="empty response"):
        asyncio.run(agent.generate(make_temporal_request()))
