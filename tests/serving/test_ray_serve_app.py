import asyncio

import pytest

from sceneops.serving.ray_serve_app import (
    NarrationRequest,
    NarrationResponse,
    NarrationService,
    build_deployment,
)


def test_ray_request_schema_excludes_simulator_state():
    request = NarrationRequest(prompt="Narrate the approved observations.")
    assert request.prompt.startswith("Narrate")
    with pytest.raises(ValueError):
        NarrationRequest.model_validate(
            {"prompt": "ok", "patient_state": {"hidden": "secret"}}
        )


def test_ray_response_schema_requires_non_empty_text():
    assert NarrationResponse(generated_text="ok").generated_text == "ok"
    with pytest.raises(ValueError):
        NarrationResponse(generated_text="")


def test_ray_dependency_is_optional_for_schema_use():
    try:
        build_deployment(generator=lambda prompt: prompt)
    except RuntimeError as exc:
        assert "Ray Serve is optional" in str(exc)


def test_ray_mocked_generation_smoke_and_timeout():
    service = NarrationService(
        lambda prompt: f"Narrated: {prompt}", timeout_seconds=1
    )
    response = asyncio.run(service.generate(NarrationRequest(prompt="approved")))
    assert response.generated_text == "Narrated: approved"

    def slow_generator(prompt: str) -> str:
        import time

        time.sleep(0.05)
        return prompt

    with_timeout = NarrationService(slow_generator, timeout_seconds=0.001)
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(with_timeout.generate(NarrationRequest(prompt="approved")))


def test_ray_model_initialization_failure_is_not_ready():
    service = NarrationService(
        None, timeout_seconds=1, initialization_error="load failed"
    )
    assert service.ready is False
    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(service.generate(NarrationRequest(prompt="approved")))
