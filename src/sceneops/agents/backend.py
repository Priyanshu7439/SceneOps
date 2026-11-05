from typing import Protocol


class TextGenerationBackend(Protocol):
    """Provider-independent async text generation contract."""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        ...
