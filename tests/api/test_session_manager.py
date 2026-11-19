import pytest

from sceneops.api.session_manager import SimulationSessionManager
from sceneops.application.service import SceneOpsApplicationService


class FakeService:
    pass


def make_service() -> SceneOpsApplicationService:
    from sceneops.agents.conversation_controller import (
        PatientConversationController,
    )
    from sceneops.agents.conversation_memory import (
        PatientConversationManager,
    )
    from sceneops.simulator.engine import SimulationEngine
    from sceneops.simulator.resources import Ambulance

    class FakePatientAgent:
        async def generate(self, request):
            raise NotImplementedError

    class FakeSafetyReviewer:
        async def review(self, request, response):
            raise NotImplementedError

    engine = SimulationEngine(ambulances=[Ambulance(ambulance_id="A001")])
    manager = PatientConversationManager()

    controller = PatientConversationController(
        patient_agent=FakePatientAgent(),
        safety_reviewer=FakeSafetyReviewer(),
        conversation_manager=manager,
    )

    return SceneOpsApplicationService(
        engine=engine,
        conversation_controller=controller,
    )


def test_create_session_returns_unique_ids():
    manager = SimulationSessionManager()

    first_id = manager.create_session(make_service)
    second_id = manager.create_session(make_service)

    assert first_id != second_id
    assert len(manager.list_sessions()) == 2


def test_get_session_returns_created_service():
    manager = SimulationSessionManager()

    session_id = manager.create_session(make_service)

    service = manager.get_session(session_id)

    assert isinstance(service, SceneOpsApplicationService)


def test_sessions_have_independent_engines():
    manager = SimulationSessionManager()

    first_id = manager.create_session(make_service)
    second_id = manager.create_session(make_service)

    first = manager.get_session(first_id)
    second = manager.get_session(second_id)

    first.start()

    assert first.engine.status.value == "running"
    assert second.engine.status.value == "created"


def test_unknown_session_raises_key_error():
    manager = SimulationSessionManager()

    with pytest.raises(KeyError, match="Unknown simulation session"):
        manager.get_session("unknown-session")


def test_only_completed_sessions_can_be_deleted():
    manager = SimulationSessionManager()
    session_id = manager.create_session(make_service)

    with pytest.raises(RuntimeError, match="Only completed"):
        manager.delete_session(session_id)

    manager.get_session(session_id).start()
    manager.get_session(session_id).complete()

    manager.delete_session(session_id)

    assert manager.list_sessions() == []


def test_invalid_factory_is_rejected():
    manager = SimulationSessionManager()

    with pytest.raises(TypeError, match="must return"):
        manager.create_session(lambda: FakeService())


def test_non_callable_factory_is_rejected():
    manager = SimulationSessionManager()

    with pytest.raises(TypeError, match="must be callable"):
        manager.create_session(None)


def test_list_sessions_returns_copy():
    manager = SimulationSessionManager()
    manager.create_session(make_service)

    session_ids = manager.list_sessions()
    session_ids.clear()

    assert len(manager.list_sessions()) == 1
