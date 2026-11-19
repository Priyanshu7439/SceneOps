from collections.abc import Callable
from uuid import uuid4

from sceneops.application.service import SceneOpsApplicationService


class SimulationSessionManager:
    """Owns isolated application services for active simulation sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SceneOpsApplicationService] = {}

    def create_session(
        self,
        service_factory: Callable[[], SceneOpsApplicationService],
    ) -> str:
        if not callable(service_factory):
            raise TypeError("service_factory must be callable.")

        service = service_factory()

        if not isinstance(service, SceneOpsApplicationService):
            raise TypeError("service_factory must return a SceneOpsApplicationService.")

        session_id = str(uuid4())
        self._sessions[session_id] = service

        return session_id

    def get_session(
        self,
        session_id: str,
    ) -> SceneOpsApplicationService:
        service = self._sessions.get(session_id)

        if service is None:
            raise KeyError(f"Unknown simulation session: {session_id}")

        return service

    def delete_session(self, session_id: str) -> None:
        service = self.get_session(session_id)

        if service.engine.status.value != "completed":
            raise RuntimeError("Only completed simulation sessions can be deleted.")

        del self._sessions[session_id]

    def list_sessions(self) -> list[str]:
        return list(self._sessions)
