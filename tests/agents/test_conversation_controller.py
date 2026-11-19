import asyncio

import pytest

from sceneops.agents.contracts import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    SafetyReview,
)
from sceneops.agents.conversation_controller import PatientConversationController
from sceneops.agents.conversation_memory import PatientConversationManager
from sceneops.agents.disclosure_rules import (
    PatientDisclosureRegistry,
    PatientQuestion,
)
from sceneops.agents.patient_dialogue import PatientDialogueAgent
from sceneops.agents.safety_reviewer import RuleBasedSafetyReviewer
from sceneops.simulator.patients import InjuryType, Patient


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
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["I feel scared."]
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        response = self.responses.pop(0)

        return AgentResponse(
            task=AgentTask.PATIENT_DIALOGUE,
            agent_name="fake_patient_agent",
            content=response,
        )


class FakeSafetyReviewer:
    def __init__(self, approved: bool = True) -> None:
        self.approved = approved
        self.reviewed_requests: list[AgentRequest] = []

    async def review(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> SafetyReview:
        self.reviewed_requests.append(request)

        return SafetyReview(
            approved=self.approved,
            reasons=() if self.approved else ("Rejected for testing.",),
        )


def make_controller(
    *,
    approved: bool = True,
    responses: list[str] | None = None,
) -> tuple[
    PatientConversationController,
    FakePatientAgent,
    PatientConversationManager,
]:
    patient_agent = FakePatientAgent(responses)
    reviewer = FakeSafetyReviewer(approved)
    manager = PatientConversationManager()

    controller = PatientConversationController(
        patient_agent=patient_agent,
        safety_reviewer=reviewer,
        conversation_manager=manager,
    )

    return controller, patient_agent, manager


def test_controller_returns_approved_patient_response():
    controller, _, manager = make_controller()

    response = asyncio.run(
        controller.handle(
            patient=make_patient(),
            user_message="How are you feeling?",
            disclosure_registry=PatientDisclosureRegistry(),
        )
    )

    assert response.content == "I feel scared."
    assert response.agent_name == "fake_patient_agent"
    assert len(manager.get_history("P001")) == 1


def test_controller_does_not_store_rejected_response():
    controller, _, manager = make_controller(
        approved=False,
        responses=["I have a serious internal injury."],
    )

    response = asyncio.run(
        controller.handle(
            patient=make_patient(),
            user_message="What is wrong?",
            disclosure_registry=PatientDisclosureRegistry(),
        )
    )

    assert response.agent_name == "safety_reviewer"
    assert manager.get_history("P001") == []


def test_controller_passes_previous_turns_to_next_request():
    controller, patient_agent, manager = make_controller(
        responses=["I feel scared.", "My leg hurts."]
    )

    async def run_two_turns():
        registry = PatientDisclosureRegistry()

        await controller.handle(
            patient=make_patient(),
            user_message="How are you feeling?",
            disclosure_registry=registry,
        )

        await controller.handle(
            patient=make_patient(),
            user_message="Can you describe it?",
            disclosure_registry=registry,
        )

    asyncio.run(run_two_turns())

    second_packet = patient_agent.requests[1].patient_packet

    assert second_packet is not None
    assert len(second_packet.conversation_memory) == 1
    assert second_packet.conversation_memory[0].user_message == "How are you feeling?"
    assert second_packet.conversation_memory[0].patient_response == "I feel scared."
    assert len(manager.get_history("P001")) == 2


def test_controller_keeps_patient_histories_separate():
    controller, _, manager = make_controller(
        responses=["I feel scared.", "I feel tired."]
    )

    async def run_two_patients():
        registry = PatientDisclosureRegistry()

        await controller.handle(
            patient=make_patient("P001"),
            user_message="How are you?",
            disclosure_registry=registry,
        )

        await controller.handle(
            patient=make_patient("P002"),
            user_message="How are you?",
            disclosure_registry=registry,
        )

    asyncio.run(run_two_patients())

    assert len(manager.get_history("P001")) == 1
    assert len(manager.get_history("P002")) == 1
    assert manager.get_history("P001")[0].patient_response == "I feel scared."
    assert manager.get_history("P002")[0].patient_response == "I feel tired."


def test_controller_rejects_mismatched_agent_response_task():
    class WrongTaskAgent:
        async def generate(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                task=AgentTask.SCENE_NARRATION,
                agent_name="wrong_agent",
                content="Rain is falling.",
            )

    manager = PatientConversationManager()
    reviewer = FakeSafetyReviewer()

    controller = PatientConversationController(
        patient_agent=WrongTaskAgent(),
        safety_reviewer=reviewer,
        conversation_manager=manager,
    )

    with pytest.raises(ValueError, match="unexpected response task"):
        asyncio.run(
            controller.handle(
                patient=make_patient(),
                user_message="How are you?",
                disclosure_registry=PatientDisclosureRegistry(),
            )
        )

    assert manager.get_history("P001") == []


def test_real_patient_agent_integrates_with_conversation_controller():
    class StyleBackend:
        async def generate(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
        ) -> str:
            return '{"style": "plain"}'

    patient = make_patient()
    registry = PatientDisclosureRegistry()
    registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    manager = PatientConversationManager()

    controller = PatientConversationController(
        patient_agent=PatientDialogueAgent(StyleBackend()),
        safety_reviewer=RuleBasedSafetyReviewer(),
        conversation_manager=manager,
    )

    async def run_conversation():
        first_response = await controller.handle(
            patient=patient,
            user_message="How much pain are you in?",
            disclosure_registry=registry,
        )

        second_response = await controller.handle(
            patient=patient,
            user_message="Do you have a headache?",
            disclosure_registry=registry,
        )

        return first_response, second_response

    first_response, second_response = asyncio.run(run_conversation())

    assert first_response.content == "My pain is around 7/10."
    assert second_response.content == "I don't know. Nobody has told me that."

    history = manager.get_history("P001")

    assert len(history) == 2
    assert history[0].patient_response == first_response.content
    assert history[1].patient_response == second_response.content


def test_controller_does_not_store_pain_inconsistent_response():
    controller, _, manager = make_controller(responses=["My pain is 9/10."])

    patient = make_patient()
    registry = PatientDisclosureRegistry()
    registry.record_answer(patient, PatientQuestion.PAIN_LEVEL)

    response = asyncio.run(
        controller.handle(
            patient=patient,
            user_message="How much pain are you in?",
            disclosure_registry=registry,
        )
    )

    assert response.agent_name == "pain_consistency_validator"
    assert manager.get_history("P001") == []
