import httpx
import pytest

from sceneops.serving.vllm_backend import VLLMBackend, VLLMBackendError


@pytest.mark.asyncio
async def test_backend_sends_openai_compatible_chat_request():
    seen = {}

    async def handler(request):
        seen["url"] = str(request.url)
        seen["json"] = await request.aread()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": " I am hurt. "}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMBackend(
        base_url="http://test/v1",
        model="sceneops-test",
        api_key="test-key",
        client=client,
    )

    result = await backend.generate(system_prompt="Act as patient", user_prompt="What happened?")

    assert result == "I am hurt."
    assert seen["url"] == "http://test/v1/chat/completions"
    assert b'"model":"sceneops-test"' in seen["json"]
    assert b'"role":"system"' in seen["json"]
    await client.aclose()


@pytest.mark.asyncio
async def test_backend_rejects_empty_prompts():
    backend = VLLMBackend(model="sceneops-test")
    with pytest.raises(ValueError, match="system_prompt"):
        await backend.generate(system_prompt=" ", user_prompt="Hello")


@pytest.mark.asyncio
async def test_backend_wraps_http_error():
    async def handler(request):
        return httpx.Response(503, json={"error": "unavailable"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMBackend(
        base_url="http://test/v1",
        model="sceneops-test",
        client=client,
    )

    with pytest.raises(VLLMBackendError, match="HTTP 503"):
        await backend.generate(system_prompt="Act", user_prompt="Hello")
    await client.aclose()


@pytest.mark.asyncio
async def test_backend_rejects_empty_model_response():
    async def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  "}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMBackend(
        base_url="http://test/v1",
        model="sceneops-test",
        client=client,
    )

    with pytest.raises(VLLMBackendError, match="empty"):
        await backend.generate(system_prompt="Act", user_prompt="Hello")
    await client.aclose()
