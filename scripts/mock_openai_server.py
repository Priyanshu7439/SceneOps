"""Local-only OpenAI-compatible mock used for application integration tests."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SceneOps local mock model")


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "sceneops-mock", "object": "model"}]}


@app.post("/v1/chat/completions")
async def completions(payload: dict) -> dict:
    messages = payload.get("messages", [])
    system = " ".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    ).casefold()
    if "patient" in system and "narrat" not in system:
        content = "I don't know. Nobody has told me that."
    else:
        content = "The approved observations were recorded for the learner."
    return {
        "id": "sceneops-mock-completion",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
