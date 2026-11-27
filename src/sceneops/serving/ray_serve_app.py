"""Optional Ray Serve adapter for simulator-grounded narration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class NarrationRequest(BaseModel):
    """Public serving payload; simulator state is never accepted here."""

    model_config = {"extra": "forbid"}
    prompt: str = Field(min_length=1, max_length=4000)


class NarrationResponse(BaseModel):
    model_config = {"extra": "forbid"}
    generated_text: str = Field(min_length=1, max_length=8000)


class HealthResponse(BaseModel):
    status: str


class NarrationService:
    """Dependency-light generation boundary used by the Ray deployment."""

    def __init__(
        self,
        generator: Callable[[str], str] | None,
        *,
        timeout_seconds: float,
        initialization_error: str | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._generator = generator
        self.timeout_seconds = timeout_seconds
        self.initialization_error = initialization_error

    @property
    def ready(self) -> bool:
        return self._generator is not None and self.initialization_error is None

    async def generate(self, request: NarrationRequest) -> NarrationResponse:
        if not self.ready:
            raise RuntimeError("Model is not initialized.")
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self._generator, request.prompt),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError("Model generation timed out.") from exc
        except Exception as exc:
            raise RuntimeError("Model generation failed.") from exc
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Model returned empty text.")
        return NarrationResponse(generated_text=text.strip())


def build_deployment(
    *,
    generator: Callable[[str], str] | None = None,
    model_id: str | None = None,
    max_new_tokens: int = 180,
    timeout_seconds: float = 60.0,
) -> Any:
    """Build a small Ray Serve deployment with injectable generation."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        from fastapi import FastAPI, HTTPException
        from ray import serve
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "Ray Serve is optional. Install ray[serve] to build this deployment."
        ) from exc

    app = FastAPI(title="SceneOps narration service")

    @serve.deployment
    @serve.ingress(app)
    class SceneOpsNarrator:
        def __init__(self) -> None:
            try:
                loaded_generator = generator or self._load_generator(
                    model_id=model_id, max_new_tokens=max_new_tokens
                )
            except Exception as exc:  # noqa: BLE001 - startup boundary
                self._service = NarrationService(
                    None,
                    timeout_seconds=timeout_seconds,
                    initialization_error=str(exc),
                )
            else:
                self._service = NarrationService(
                    loaded_generator,
                    timeout_seconds=timeout_seconds,
                )

        @staticmethod
        def _load_generator(
            *, model_id: str | None, max_new_tokens: int
        ) -> Callable[[str], str]:
            if not model_id:
                raise ValueError("model_id is required when generator is not injected")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Install torch and transformers for the built-in generator."
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

            def generate(prompt: str) -> str:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = outputs[0][inputs["input_ids"].shape[1] :]
                return tokenizer.decode(generated, skip_special_tokens=True).strip()

            return generate

        @app.get("/health", response_model=HealthResponse)
        async def health(self) -> HealthResponse:
            if not self._service.ready:
                raise HTTPException(status_code=503, detail="Model is not ready")
            return HealthResponse(status="ok")

        @app.post("/generate", response_model=NarrationResponse)
        async def generate(self, request: NarrationRequest) -> NarrationResponse:
            try:
                return await self._service.generate(request)
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SceneOpsNarrator


def serve_model(*, model_id: str, address: str = "127.0.0.1:8000") -> None:
    """Start the optional deployment from an explicit serving command."""

    from ray import serve

    deployment = build_deployment(model_id=model_id)
    host, port = address.rsplit(":", 1)
    serve.start(http_options={"host": host, "port": int(port)})
    serve.run(deployment.bind(), name="sceneops-narrator")
