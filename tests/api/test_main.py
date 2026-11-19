from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from sceneops.api.main import create_app
from sceneops.api.session_manager import SimulationSessionManager
from sceneops.serving.vllm_backend import VLLMBackendError


class FakeVLLMBackend:
    response = "I feel scared."

    received_clients: ClassVar[list] = []

    def __init__(self, *, base_url, model, api_key=None, client=None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.client = client
        type(self).received_clients.append(client)

    async def generate(self, *, system_prompt, user_prompt):
        return self.response


class FailingVLLMBackend(FakeVLLMBackend):
    async def generate(self, *, system_prompt, user_prompt):
        raise VLLMBackendError("vLLM request timed out.")


@pytest.fixture
def client_and_manager():
    manager = SimulationSessionManager()
    app = create_app(session_manager=manager)

    with TestClient(app) as client:
        yield client, manager


def create_session(client: TestClient) -> str:
    response = client.post(
        "/sessions",
        json={"scenario_index": 1},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def install_backend(monkeypatch, backend_class):
    monkeypatch.setattr(
        "sceneops.api.main.VLLMBackend",
        backend_class,
    )


def test_health_endpoint(client_and_manager):
    client, _ = client_and_manager

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_create_session(client_and_manager):
    client, manager = client_and_manager

    response = client.post("/sessions", json={})

    assert response.status_code == 201
    assert response.json()["scenario_id"] == "MCI-001"
    assert response.json()["patient_count"] == 20
    assert response.json()["status"] == "created"
    assert len(manager.list_sessions()) == 1


def test_create_session_with_invalid_index(client_and_manager):
    client, _ = client_and_manager

    response = client.post(
        "/sessions",
        json={"scenario_index": 0},
    )

    assert response.status_code == 422


def test_get_session_snapshot(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.get(f"/sessions/{session_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "created"
    assert response.json()["patient_count"] == 20


def test_unknown_session_returns_404(client_and_manager):
    client, _ = client_and_manager

    response = client.get("/sessions/unknown")

    assert response.status_code == 404


def test_session_lifecycle(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    start = client.post(f"/sessions/{session_id}/start")
    assert start.status_code == 200
    assert start.json()["status"] == "running"

    pause = client.post(f"/sessions/{session_id}/pause")
    assert pause.status_code == 200
    assert pause.json()["status"] == "paused"

    resume = client.post(f"/sessions/{session_id}/resume")
    assert resume.status_code == 200
    assert resume.json()["status"] == "running"

    complete = client.post(f"/sessions/{session_id}/complete")
    assert complete.status_code == 200
    assert complete.json()["status"] == "completed"


def test_invalid_lifecycle_transition_returns_409(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(f"/sessions/{session_id}/pause")

    assert response.status_code == 409


def test_advance_simulation_clock(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/advance",
        json={"seconds": 120},
    )

    assert response.status_code == 200
    assert response.json()["elapsed_seconds"] == 120
    assert response.json()["tick_count"] == 1


def test_advance_clock_before_start_returns_409(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(
        f"/sessions/{session_id}/advance",
        json={"seconds": 60},
    )

    assert response.status_code == 409


def test_advance_clock_rejects_invalid_seconds(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/advance",
        json={"seconds": 0},
    )

    assert response.status_code == 422


def test_record_patient_question(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/questions",
        json={"question": "pain_level"},
    )

    assert response.status_code == 200
    assert response.json()["fact"] == "reported_pain_level"
    assert response.json()["value"].endswith("/10")


def test_unknown_patient_question_returns_404(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/UNKNOWN/questions",
        json={"question": "pain_level"},
    )

    assert response.status_code == 404


def test_dialogue_before_start_returns_409(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(
        f"/sessions/{session_id}/patients/P001/dialogue",
        json={"message": "How are you feeling?"},
    )

    assert response.status_code == 409


def test_patient_dialogue_success(monkeypatch, client_and_manager):
    FakeVLLMBackend.received_clients.clear()
    install_backend(monkeypatch, FakeVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/dialogue",
        json={"message": "How are you feeling?"},
    )

    assert response.status_code == 200
    assert response.json()["patient_id"] == "P001"
    assert response.json()["agent_name"] == "patient_dialogue_agent"
    assert response.json()["content"] == ("I don't know. Nobody has told me that.")


def test_patient_dialogue_vllm_failure_returns_503(
    monkeypatch,
    client_and_manager,
):
    install_backend(monkeypatch, FailingVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/dialogue",
        json={"message": "How are you feeling?"},
    )

    assert response.status_code == 503
    assert "timed out" in response.json()["detail"]


def test_patient_dialogue_rejects_unsafe_response(
    monkeypatch,
    client_and_manager,
):
    class UnsafeVLLMBackend(FakeVLLMBackend):
        response = "You have a heart attack."

    install_backend(monkeypatch, UnsafeVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/dialogue",
        json={"message": "How are you feeling?"},
    )

    assert response.status_code == 200
    assert "heart attack" not in response.json()["content"].casefold()


def test_multiple_sessions_share_http_client(
    monkeypatch,
    client_and_manager,
):
    FakeVLLMBackend.received_clients.clear()
    install_backend(monkeypatch, FakeVLLMBackend)

    client, _ = client_and_manager

    first_session = create_session(client)
    second_session = create_session(client)

    assert first_session != second_session
    assert len(FakeVLLMBackend.received_clients) == 2

    first_client, second_client = FakeVLLMBackend.received_clients

    assert first_client is second_client
    assert first_client is not None
    assert first_client is client.app.state.http_client


def test_http_client_closes_on_application_shutdown():
    app = create_app()

    with TestClient(app):
        shared_client = app.state.http_client
        assert not shared_client.is_closed

    assert shared_client.is_closed


def test_delete_completed_session(client_and_manager):
    client, manager = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")
    client.post(f"/sessions/{session_id}/complete")

    response = client.delete(f"/sessions/{session_id}")

    assert response.status_code == 200
    assert session_id not in manager.list_sessions()


def test_running_session_cannot_be_deleted(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.delete(f"/sessions/{session_id}")

    assert response.status_code == 409


def test_patient_observation_success(client_and_manager):
    client, manager = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    service = manager.get_session(session_id)
    expected_rate = service.engine.patients["P001"].respiratory_rate

    response = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "respiratory_rate"},
    )

    assert response.status_code == 201

    data = response.json()
    assert data["patient_id"] == "P001"
    assert data["observation_type"] == "respiratory_rate"
    assert data["value"] == str(expected_rate)
    assert data["observed_at_seconds"] == 0

    recorded = service.engine.observation_registry.get_for_patient("P001")
    assert len(recorded) == 1


def test_patient_observation_before_start_returns_409(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "respiratory_rate"},
    )

    assert response.status_code == 409


def test_patient_observation_while_paused_returns_409(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")
    client.post(f"/sessions/{session_id}/pause")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "respiratory_rate"},
    )

    assert response.status_code == 409


def test_unknown_patient_observation_returns_404(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/UNKNOWN/observations",
        json={"observation_type": "respiratory_rate"},
    )

    assert response.status_code == 404


def test_unknown_session_observation_returns_404(client_and_manager):
    client, _ = client_and_manager

    response = client.post(
        "/sessions/unknown/patients/P001/observations",
        json={"observation_type": "respiratory_rate"},
    )

    assert response.status_code == 404


def test_unsupported_patient_observation_returns_422(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "reported_pain"},
    )

    assert response.status_code == 422


def test_invalid_observation_type_returns_422(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "invalid_type"},
    )

    assert response.status_code == 422


def test_patient_observation_rejects_client_supplied_value(client_and_manager):
    client, manager = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={
            "observation_type": "respiratory_rate",
            "value": "999",
        },
    )

    assert response.status_code == 422

    service = manager.get_session(session_id)
    recorded = service.engine.observation_registry.get_for_patient("P001")
    assert recorded == []


def test_scene_narration_success(monkeypatch, client_and_manager):
    install_backend(monkeypatch, FakeVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    observation = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "respiratory_rate"},
    )
    assert observation.status_code == 201

    response = client.post(
        f"/sessions/{session_id}/narration",
        json={"message": "Summarize the observations recorded so far."},
    )

    assert response.status_code == 200
    assert response.json()["agent_name"] == "scene_narration_agent"
    assert response.json()["content"] == "I feel scared."


def test_scene_narration_before_start_returns_409(monkeypatch, client_and_manager):
    install_backend(monkeypatch, FakeVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(
        f"/sessions/{session_id}/narration",
        json={"message": "Summarize the scene."},
    )

    assert response.status_code == 409


def test_scene_narration_without_observations_returns_422(
    monkeypatch, client_and_manager
):
    install_backend(monkeypatch, FakeVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    response = client.post(
        f"/sessions/{session_id}/narration",
        json={"message": "Summarize the scene."},
    )

    assert response.status_code == 422


def test_scene_narration_unknown_session_returns_404(monkeypatch, client_and_manager):
    install_backend(monkeypatch, FakeVLLMBackend)

    client, _ = client_and_manager

    response = client.post(
        "/sessions/unknown/narration",
        json={"message": "Summarize the scene."},
    )

    assert response.status_code == 404


def test_scene_narration_rejects_invalid_request(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(
        f"/sessions/{session_id}/narration",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_scene_narration_rejects_client_supplied_facts(client_and_manager):
    client, _ = client_and_manager
    session_id = create_session(client)

    response = client.post(
        f"/sessions/{session_id}/narration",
        json={
            "message": "Summarize the scene.",
            "scene_facts": ["P001 has a respiratory rate of 999"],
        },
    )

    assert response.status_code == 422


def test_scene_narration_vllm_failure_returns_503(monkeypatch, client_and_manager):
    install_backend(monkeypatch, FailingVLLMBackend)

    client, _ = client_and_manager
    session_id = create_session(client)

    client.post(f"/sessions/{session_id}/start")

    observation = client.post(
        f"/sessions/{session_id}/patients/P001/observations",
        json={"observation_type": "respiratory_rate"},
    )
    assert observation.status_code == 201

    response = client.post(
        f"/sessions/{session_id}/narration",
        json={"message": "Summarize the observations."},
    )

    assert response.status_code == 503
    assert "timed out" in response.json()["detail"]
