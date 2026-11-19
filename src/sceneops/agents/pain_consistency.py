from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


class PainState(BaseModel):
    """An explicitly authorized pain score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: int = Field(ge=0, le=10)


class PainConsistencyResult(BaseModel):
    """Result of checking generated dialogue against authorized pain data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    reasons: tuple[str, ...] = ()


class PainConsistencyValidator:
    """Validate explicit numeric pain claims against authorized disclosure.

    This validator does not infer pain changes from conversation history
    or access hidden simulator state.
    """

    _PAIN_SCORE_PATTERN = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10\b",
        re.IGNORECASE,
    )

    def validate(
        self,
        *,
        response: str,
        current_pain: PainState | None,
    ) -> PainConsistencyResult:
        reasons: list[str] = []

        matches = self._PAIN_SCORE_PATTERN.findall(response)

        if matches and current_pain is None:
            reasons.append(
                "Response contains a pain score without an authorized pain disclosure."
            )

        elif current_pain is not None:
            for match in matches:
                reported_value = float(match)

                if reported_value != current_pain.value:
                    reasons.append(
                        f"Response reports pain {reported_value}/10, "
                        f"but the latest authorized disclosure is "
                        f"{current_pain.value}/10."
                    )

        return PainConsistencyResult(
            approved=not reasons,
            reasons=tuple(reasons),
        )
