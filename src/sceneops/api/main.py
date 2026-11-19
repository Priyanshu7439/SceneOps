import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status

from sceneops.agents.conversation_controller import (
    PatientConversationController,
)
from sceneops.agents.conversation_memory import PatientConversationManager
from sceneops.agents.orchestrator import AgentOrchestrator
from sceneops.agents.patient_dialogue import PatientDialogueAgent
from sceneops.agents.safety_reviewer import RuleBasedSafetyReviewer
from sceneops.agents.scene_narration import SceneNarrationAgent
from sceneops.api.schemas import (
    AdvanceTimeRequest,
    ClockResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    MessageResponse,
    PatientDialogueRequest,
    PatientDialogueResponse,
    PatientObservationRequest,
    PatientObservationResponse,
    PatientQuestionRequest,
    PatientQuestionResponse,
    SceneNarrationRequest,
    SceneNarrationResponse,
    SessionSnapshotResponse,
)
from sceneops.api.session_manager import SimulationSessionManager
from sceneops.application.service import SceneOpsApplicationService
from sceneops.serving.vllm_backend import VLLMBackend, VLLMBackendError
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.simulator.scenario_factory import create_mass_casualty_scenario


def create_app(
    session_manager: SimulationSessionManager | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            app.state.http_client = client
            yield

    app = FastAPI(
        title="SceneOps API",
        description=(
            "Educational mass-casualty triage simulation API. "
            "This simulator is not a clinical decision-support system."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    manager = session_manager or SimulationSessionManager()
    app.state.session_manager = manager

    vllm_base_url = os.getenv(
        "SCENEOPS_VLLM_BASE_URL",
        "http://localhost:8000/v1",
    )
    vllm_model = os.getenv(
        "SCENEOPS_VLLM_MODEL",
        "local-model",
    )
    vllm_api_key = os.getenv("SCENEOPS_VLLM_API_KEY")

    def get_service(session_id: str) -> SceneOpsApplicationService:
        try:
            return manager.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "sceneops-api"}

    @app.post(
        "/sessions",
        response_model=CreateSessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(
        request: CreateSessionRequest,
    ) -> CreateSessionResponse:
        scenario = create_mass_casualty_scenario(request.scenario_index)
        engine = ScenarioBuilder.build(scenario)

        backend = VLLMBackend(
            base_url=vllm_base_url,
            model=vllm_model,
            api_key=vllm_api_key,
            client=app.state.http_client,
        )

        controller = PatientConversationController(
            patient_agent=PatientDialogueAgent(backend),
            safety_reviewer=RuleBasedSafetyReviewer(),
            conversation_manager=PatientConversationManager(),
        )

        scene_orchestrator = AgentOrchestrator(
            patient_agent=PatientDialogueAgent(backend),
            scene_agent=SceneNarrationAgent(backend),
            safety_reviewer=RuleBasedSafetyReviewer(),
        )

        service = SceneOpsApplicationService(
            engine=engine,
            conversation_controller=controller,
            agent_orchestrator=scene_orchestrator,
        )

        session_id = manager.create_session(lambda: service)

        return CreateSessionResponse(
            session_id=session_id,
            scenario_id=scenario.scenario_id,
            title=scenario.title,
            description=scenario.description,
            patient_count=len(scenario.patients),
            status=engine.status.value,
        )

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionSnapshotResponse,
    )
    async def get_session(
        session_id: str,
    ) -> SessionSnapshotResponse:
        service = get_service(session_id)
        snapshot = service.get_snapshot()

        return SessionSnapshotResponse(
            session_id=session_id,
            **snapshot,
        )

    @app.post(
        "/sessions/{session_id}/start",
        response_model=MessageResponse,
    )
    async def start_session(session_id: str) -> MessageResponse:
        service = get_service(session_id)

        try:
            service.start()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return MessageResponse(
            message="Simulation started.",
            status=service.engine.status.value,
        )

    @app.post(
        "/sessions/{session_id}/pause",
        response_model=MessageResponse,
    )
    async def pause_session(session_id: str) -> MessageResponse:
        service = get_service(session_id)

        try:
            service.pause()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return MessageResponse(
            message="Simulation paused.",
            status=service.engine.status.value,
        )

    @app.post(
        "/sessions/{session_id}/resume",
        response_model=MessageResponse,
    )
    async def resume_session(session_id: str) -> MessageResponse:
        service = get_service(session_id)

        try:
            service.resume()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return MessageResponse(
            message="Simulation resumed.",
            status=service.engine.status.value,
        )

    @app.post(
        "/sessions/{session_id}/complete",
        response_model=MessageResponse,
    )
    async def complete_session(session_id: str) -> MessageResponse:
        service = get_service(session_id)

        try:
            service.complete()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return MessageResponse(
            message="Simulation completed.",
            status=service.engine.status.value,
        )

    @app.delete(
        "/sessions/{session_id}",
        response_model=MessageResponse,
    )
    async def delete_session(session_id: str) -> MessageResponse:
        get_service(session_id)

        try:
            manager.delete_session(session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return MessageResponse(
            message="Simulation session deleted.",
            status="deleted",
        )

    @app.post(
        "/sessions/{session_id}/advance",
        response_model=ClockResponse,
    )
    async def advance_simulation(
        session_id: str,
        request: AdvanceTimeRequest,
    ) -> ClockResponse:
        service = get_service(session_id)

        try:
            snapshot = service.advance_time(request.seconds)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return ClockResponse(
            elapsed_seconds=snapshot.elapsed_seconds,
            tick_count=snapshot.tick_count,
        )

    @app.post(
        "/sessions/{session_id}/patients/{patient_id}/dialogue",
        response_model=PatientDialogueResponse,
    )
    async def patient_dialogue(
        session_id: str,
        patient_id: str,
        request: PatientDialogueRequest,
    ) -> PatientDialogueResponse:
        service = get_service(session_id)

        try:
            response = await service.ask_patient(
                patient_id,
                request.message,
                disclose_age=request.disclose_age,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except VLLMBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return PatientDialogueResponse(
            patient_id=patient_id,
            agent_name=response.agent_name,
            content=response.content,
        )

    @app.post(
        "/sessions/{session_id}/patients/{patient_id}/observations",
        response_model=PatientObservationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def observe_patient(
        session_id: str,
        patient_id: str,
        request: PatientObservationRequest,
    ) -> PatientObservationResponse:
        service = get_service(session_id)

        try:
            observation = service.observe_patient(
                patient_id,
                request.observation_type,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except TypeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        return PatientObservationResponse(
            patient_id=observation.patient_id,
            observation_type=observation.observation_type,
            value=observation.value,
            observed_at_seconds=observation.observed_at_seconds,
        )

    @app.post(
        "/sessions/{session_id}/narration",
        response_model=SceneNarrationResponse,
    )
    async def narrate_scene(
        session_id: str,
        request: SceneNarrationRequest,
    ) -> SceneNarrationResponse:
        service = get_service(session_id)

        try:
            response = await service.narrate_scene(request.message)
        except VLLMBackendError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        return SceneNarrationResponse(
            agent_name=response.agent_name,
            content=response.content,
        )

    @app.post(
        "/sessions/{session_id}/patients/{patient_id}/questions",
        response_model=PatientQuestionResponse,
    )
    async def record_patient_question(
        session_id: str,
        patient_id: str,
        request: PatientQuestionRequest,
    ) -> PatientQuestionResponse:
        service = get_service(session_id)

        try:
            record = service.record_patient_question(
                patient_id,
                request.question,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return PatientQuestionResponse(
            patient_id=record.patient_id,
            fact=record.disclosure.fact,
            value=record.disclosure.value,
        )

    return app


app = create_app()
