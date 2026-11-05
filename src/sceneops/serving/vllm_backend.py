from __future__ import annotations

import httpx


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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        client = self._client
        owns_temporary_client = client is None
        if owns_temporary_client:
            client = httpx.AsyncClient(timeout=self.timeout_seconds)

        try:
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
            raise VLLMBackendError("vLLM request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise VLLMBackendError(
                f"vLLM returned HTTP {status}."
            ) from exc
        except (httpx.RequestError, KeyError, IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, VLLMBackendError):
                raise
            raise VLLMBackendError(
                "Could not complete or parse the vLLM response."
            ) from exc
        finally:
            if owns_temporary_client:
                await client.aclose()

    async def aclose(self) -> None:
        """Close an injected client only when this backend owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
