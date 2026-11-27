from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class VLLMBackendError(RuntimeError):
    """Raised when the vLLM-compatible generation endpoint fails."""


class VLLMBackend:
    """Async client for vLLM's OpenAI-compatible chat completions API.

    The API key is optional for local development. Keep it in an environment
    variable or secret manager; never hard-code it in source code.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        temperature: float = 0.3,
        max_tokens: int = 300,
        auto_discover_model: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be empty.")
        if not model.strip():
            raise ValueError("model cannot be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.auto_discover_model = auto_discover_model
        self._resolved_model: str | None = None
        self._client = client
        self._owns_client = client is None

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty.")
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty.")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client = self._client
        owns_temporary_client = client is None
        if owns_temporary_client:
            client = httpx.AsyncClient(timeout=self.timeout_seconds)

        try:
            model = await self._resolve_model(client, headers)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise VLLMBackendError("Model returned empty or invalid content.")
            return content.strip()
        except httpx.TimeoutException as exc:
            raise VLLMBackendError(
                f"Model provider request timed out at {self.base_url}."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise VLLMBackendError(
                f"Model provider returned HTTP {status} from {exc.request.url.path}."
            ) from exc
        except httpx.RequestError as exc:
            raise VLLMBackendError(
                f"Could not reach model provider at {self.base_url}: "
                f"{type(exc).__name__}."
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, VLLMBackendError):
                raise
            raise VLLMBackendError(
                "Model provider returned an invalid chat completion schema "
                "(expected choices[0].message.content)."
            ) from exc
        finally:
            if owns_temporary_client:
                await client.aclose()

    async def _resolve_model(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> str:
        if not self.auto_discover_model:
            return self.model
        if self._resolved_model is not None:
            return self._resolved_model

        response = await client.get(
            f"{self.base_url}/models",
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        model_ids: list[str] = []
        for item in data.get("data", []):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
                model_ids.extend(
                    alias for alias in item.get("aliases", []) if isinstance(alias, str)
                )
        for item in data.get("models", []):
            if isinstance(item, dict):
                for key in ("id", "model", "name"):
                    if isinstance(item.get(key), str):
                        model_ids.append(item[key])

        unique_ids = list(dict.fromkeys(model_ids))
        if self.model in unique_ids:
            self._resolved_model = self.model
            return self._resolved_model
        if len(unique_ids) == 1:
            self._resolved_model = unique_ids[0]
            logger.warning(
                "Configured model %r was not advertised; using the provider's "
                "only advertised model %r.",
                self.model,
                self._resolved_model,
            )
            return self._resolved_model
        available = ", ".join(unique_ids) or "none"
        raise VLLMBackendError(
            f"Configured model {self.model!r} was not advertised by the model "
            f"provider. Available models: {available}."
        )

    async def aclose(self) -> None:
        """Close an injected client only when this backend owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
