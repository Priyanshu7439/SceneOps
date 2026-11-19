import asyncio

from sceneops.agents.contracts import AgentRequest, AgentResponse, AgentTask
from sceneops.agents.disclosure import (
    ConversationTurn,
    DisclosureFact,
    DisclosurePacket,
    PatientProfile,
)
from sceneops.agents.safety_reviewer import RuleBasedSafetyReviewer


def make_request(
    *,
    disclosures: list[DisclosureFact] | None = None,
    history: list[ConversationTurn] | None = None,
) -> AgentRequest:
    packet = DisclosurePacket(
        patient_id="P001",
        profile=PatientProfile(
            personality="anxious",
            speech_style="plain",
        ),
        current_disclosures=disclosures or [],
        conversation_memory=history or [],
    )

    return AgentRequest(
        task=AgentTask.PATIENT_DIALOGUE,
        user_message="How are you feeling?",
        patient_packet=packet,
    )


def make_response(content: str) -> AgentResponse:
    return AgentResponse(
        task=AgentTask.PATIENT_DIALOGUE,
        agent_name="test_patient_agent",
        content=content,
    )


def review(content: str, request: AgentRequest):
    reviewer = RuleBasedSafetyReviewer()
    return asyncio.run(reviewer.review(request, make_response(content)))


def test_approves_response_using_disclosed_pain():
    request = make_request(
        disclosures=[
            DisclosureFact(
                fact="reported_pain_level",
                value="7/10",
            )
        ]
    )

    result = review("My pain is around 7/10.", request)

    assert result.approved is True
    assert result.reasons == ()


def test_rejects_unsupported_respiratory_rate():
    request = make_request(
        disclosures=[
            DisclosureFact(
                fact="reported_pain_level",
                value="7/10",
            )
        ]
    )

    result = review("My respiratory rate is 24.", request)

    assert result.approved is False
    assert any(
        "undisclosed clinical measurement" in reason for reason in result.reasons
    )


def test_rejects_diagnosis_language():
    request = make_request()

    result = review("You have internal bleeding.", request)

    assert result.approved is False
    assert result.reasons


def test_approves_simple_unknown_response():
    request = make_request()

    result = review("I don't know. Nobody has told me.", request)

    assert result.approved is True


def test_non_empty_conversation_history_does_not_crash():
    request = make_request(
        history=[
            ConversationTurn(
                user_message="How are you feeling?",
                patient_response="I feel scared.",
            )
        ]
    )

    result = review("I feel scared.", request)

    assert result.approved is True


def test_conversation_history_cannot_authorize_undisclosed_measurement():
    request = make_request(
        history=[
            ConversationTurn(
                user_message="What is your breathing rate?",
                patient_response="My respiratory rate is 24.",
            )
        ]
    )

    result = review("My respiratory rate is 24.", request)

    assert result.approved is False
    assert any(
        "undisclosed clinical measurement" in reason for reason in result.reasons
    )


def test_reviewer_rejects_paraphrased_diagnosis():
    request = make_request()

    result = review(
        "There may be internal bleeding.",
        request,
    )

    assert result.approved is False
    assert any("diagnosis" in reason.casefold() for reason in result.reasons)


def make_scene_request(facts):
    return AgentRequest(
        task=AgentTask.SCENE_NARRATION,
        user_message="Describe the current scene.",
        scene_facts=tuple(facts),
    )


def make_scene_response(content: str) -> AgentResponse:
    return AgentResponse(
        task=AgentTask.SCENE_NARRATION,
        agent_name="test_scene_agent",
        content=content,
    )


def review_scene(content: str, request: AgentRequest):
    reviewer = RuleBasedSafetyReviewer()
    return asyncio.run(reviewer.review(request, make_scene_response(content)))


def test_scene_narration_accepts_correct_patient_measurement():
    from sceneops.agents.contracts import SceneFact

    request = make_scene_request(
        [
            SceneFact(
                fact="patient_observation:P001:respiratory_rate",
                value="20 (observed at simulation second 60)",
            )
        ]
    )

    result = review_scene("P001 has a respiratory rate of 20.", request)

    assert result.approved is True


def test_scene_narration_rejects_measurement_assigned_to_wrong_patient():
    from sceneops.agents.contracts import SceneFact

    request = make_scene_request(
        [
            SceneFact(
                fact="patient_observation:P001:respiratory_rate",
                value="20 (observed at simulation second 60)",
            ),
            SceneFact(
                fact="patient_observation:P002:respiratory_rate",
                value="30 (observed at simulation second 60)",
            ),
        ]
    )

    result = review_scene("P002 has a respiratory rate of 20.", request)

    assert result.approved is False


def test_scene_narration_rejects_unsupported_measurement():
    from sceneops.agents.contracts import SceneFact

    request = make_scene_request(
        [
            SceneFact(
                fact="patient_observation:P001:respiratory_rate",
                value="20 (observed at simulation second 60)",
            )
        ]
    )

    result = review_scene("P001 has a respiratory rate of 24.", request)

    assert result.approved is False


def test_scene_narration_rejects_wrong_observation_type():
    from sceneops.agents.contracts import SceneFact

    request = make_scene_request(
        [
            SceneFact(
                fact="patient_observation:P001:mental_status",
                value="alert (observed at simulation second 60)",
            )
        ]
    )

    result = review_scene("P001 has a respiratory rate of 20.", request)

    assert result.approved is False
