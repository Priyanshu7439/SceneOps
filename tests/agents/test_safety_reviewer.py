import asyncio

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    TemporalObservation,
)
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

def make_temporal_request() -> AgentRequest:
    return AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe the recorded changes.",
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
                observation_type="respiratory_rate",
                value="30",
                observed_at_seconds=60,
            ),
        ),
    )


def review_temporal(content: str, request: AgentRequest):
    reviewer = RuleBasedSafetyReviewer()
    response = AgentResponse(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        agent_name="test_temporal_agent",
        content=content,
    )
    return asyncio.run(reviewer.review(request, response))


def test_temporal_reviewer_accepts_matching_patient_value_and_timestamp():
    request = make_temporal_request()

    result = review_temporal(
        "At simulation second 60, P001's respiratory rate was 24 breaths per minute.",
        request,
    )

    assert result.approved is True


def test_temporal_reviewer_rejects_unsupported_measurement_value():
    request = make_temporal_request()

    result = review_temporal(
        "At simulation second 60, P001's respiratory rate was 28 breaths per minute.",
        request,
    )

    assert result.approved is False


def test_temporal_reviewer_rejects_measurement_assigned_to_wrong_patient():
    request = make_temporal_request()

    result = review_temporal(
        "At simulation second 60, P002's respiratory rate was 24 breaths per minute.",
        request,
    )

    assert result.approved is False


def test_temporal_reviewer_rejects_measurement_with_wrong_timestamp():
    request = make_temporal_request()

    result = review_temporal(
        "At simulation second 0, P001's respiratory rate was 24 breaths per minute.",
        request,
    )

    assert result.approved is False

import pytest


@pytest.mark.parametrize(
    ("observation_type", "value", "claim"),
    [
        (
            "radial_pulse",
            "absent",
            "radial pulse was absent",
        ),
        (
            "mental_status",
            "confused",
            "mental status was confused",
        ),
        (
            "ambulatory_status",
            "non_ambulatory",
            "ambulatory status was non_ambulatory",
        ),
        (
            "visible_hemorrhage",
            "visible major hemorrhage",
            "visible major hemorrhage was recorded",
        ),
        (
            "follows_commands",
            "does not follow commands",
            "P001 does not follow commands",
        ),
    ],
)
def test_temporal_reviewer_accepts_recorded_categorical_observations(
    observation_type,
    value,
    claim,
):
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type=observation_type,
                value=value,
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        f"At simulation second 60, P001: {claim}.",
        request,
    )

    assert result.approved is True


def test_temporal_reviewer_rejects_wrong_categorical_value():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="mental_status",
                value="confused",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        "At simulation second 60, P001's mental status was alert.",
        request,
    )

    assert result.approved is False

def test_temporal_reviewer_rejects_measurement_without_patient_id():
    request = make_temporal_request()

    result = review_temporal(
        "At simulation second 60, respiratory rate was 24 breaths per minute.",
        request,
    )

    assert result.approved is False


def test_temporal_reviewer_rejects_measurement_without_timestamp():
    request = make_temporal_request()

    result = review_temporal(
        "P001's respiratory rate was 24 breaths per minute.",
        request,
    )

    assert result.approved is False


def test_temporal_reviewer_rejects_wrong_observation_type():
    request = make_temporal_request()

    result = review_temporal(
        "At simulation second 60, P001's radial pulse was 24 bpm.",
        request,
    )

    assert result.approved is False


def test_temporal_reviewer_rejects_contradictory_hemorrhage_claim():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="visible_hemorrhage",
                value="no visible major hemorrhage",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        "At simulation second 60, P001 had visible major hemorrhage.",
        request,
    )

    assert result.approved is False


def make_same_value_temporal_request(
    *,
    second_timestamp: int = 60,
) -> AgentRequest:
    return AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Summarize the chronological observations.",
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
                value="20",
                observed_at_seconds=second_timestamp,
            ),
        ),
    )


def test_temporal_reviewer_accepts_plural_timestamp_range_with_same_value():
    request = make_same_value_temporal_request()

    result = review_temporal(
        (
            "Between simulation seconds 0 and 60, no changes were recorded "
            "in the observed fields for patient P001. The recorded "
            "respiratory rate remained at 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is True
    assert result.reasons == ()


def test_temporal_reviewer_rejects_timestamp_range_if_one_timestamp_is_unsupported():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Summarize the chronological observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
        ),
    )

    result = review_temporal(
        (
            "Between simulation seconds 0 and 60, patient P001's recorded "
            "respiratory rate remained at 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False
    assert any(
        "timestamp" in reason.casefold()
        or "does not match" in reason.casefold()
        for reason in result.reasons
    )


def test_temporal_reviewer_rejects_plural_timestamp_range_for_wrong_patient():
    request = make_same_value_temporal_request()

    result = review_temporal(
        (
            "Between simulation seconds 0 and 60, patient P002's "
            "respiratory rate remained at 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False


def test_temporal_reviewer_rejects_plural_timestamp_range_with_wrong_value():
    request = make_same_value_temporal_request()

    result = review_temporal(
        (
            "Between simulation seconds 0 and 60, patient P001's "
            "respiratory rate remained at 22 breaths per minute."
        ),
        request,
    )

    assert result.approved is False




def test_temporal_reviewer_accepts_each_changing_measurement_at_its_own_timestamp():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Summarize the chronological observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=60,
            ),
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="35",
                observed_at_seconds=120,
            ),
        ),
    )

    result = review_temporal(
        (
            "At simulation second 0, Patient P005: respiratory rate is 20 "
            "breaths per minute. "
            "At simulation second 60, Patient P005: respiratory rate is 20 "
            "breaths per minute. "
            "At simulation second 120, Patient P005: respiratory rate is 35 "
            "breaths per minute."
        ),
        request,
    )

    assert result.approved is True
    assert result.reasons == ()


def test_temporal_reviewer_rejects_changing_value_attached_to_ambiguous_range():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Summarize the chronological observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="35",
                observed_at_seconds=120,
            ),
        ),
    )

    result = review_temporal(
        (
            "Between simulation seconds 0 and 120, Patient P005's recorded "
            "respiratory rate was 35 breaths per minute."
        ),
        request,
    )

    assert result.approved is False
    assert any("does not match" in reason.casefold() for reason in result.reasons)
def test_temporal_reviewer_accepts_no_visible_hemorrhage():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="visible_hemorrhage",
                value="no visible major hemorrhage",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        "At simulation second 60, P001 has no visible major hemorrhage.",
        request,
    )

    assert result.approved is True

def test_temporal_reviewer_rejects_ambiguous_patient_attribution():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
            TemporalObservation(
                patient_id="P002",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        (
            "At simulation second 0, P001 respiratory rate is "
            "20 breaths per minute. "
            "At simulation second 60, P002 is also present. "
            "Respiratory rate is 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False

def test_temporal_reviewer_rejects_measurement_with_multiple_patient_ids():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P002",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        (
            "At simulation second 60, P001 and P002 have "
            "a respiratory rate of 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False

def test_temporal_reviewer_rejects_stale_timestamp_reuse():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
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
                value="30",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        (
            "At simulation second 0, P001 respiratory rate is "
            "20 breaths per minute. "
            "At simulation second 60, respiratory rate is "
            "20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False

def test_temporal_reviewer_rejects_single_value_across_changing_timestamp_range():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
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
                value="30",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        (
            "Between simulation seconds 0 and 60, P001's "
            "respiratory rate remained at 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False

def test_temporal_reviewer_rejects_stale_categorical_observation():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="mental_status",
                value="alert",
                observed_at_seconds=0,
            ),
            TemporalObservation(
                patient_id="P001",
                observation_type="mental_status",
                value="unresponsive",
                observed_at_seconds=60,
            ),
        ),
    )

    result = review_temporal(
        (
            "At simulation second 0, P001 mental status was alert. "
            "At simulation second 60, P001 mental status was alert."
        ),
        request,
    )

    assert result.approved is False

def test_temporal_reviewer_rejects_measurement_without_own_timestamp():
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Describe observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P001",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
        ),
    )

    result = review_temporal(
        (
            "At simulation second 0, P001 respiratory rate is "
            "20 breaths per minute. "
            "P001 respiratory rate is 20 breaths per minute."
        ),
        request,
    )

    assert result.approved is False