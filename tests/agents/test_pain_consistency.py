import pytest

from sceneops.agents.pain_consistency import (
    PainConsistencyValidator,
    PainState,
)


@pytest.fixture
def validator() -> PainConsistencyValidator:
    return PainConsistencyValidator()


def test_accepts_matching_authorized_pain_score(validator):
    result = validator.validate(
        response="My pain is around 6/10.",
        current_pain=PainState(value=6),
    )

    assert result.approved is True
    assert result.reasons == ()


def test_rejects_mismatched_pain_score(validator):
    result = validator.validate(
        response="My pain has increased to 9/10.",
        current_pain=PainState(value=6),
    )

    assert result.approved is False
    assert any("9.0/10" in reason for reason in result.reasons)


def test_accepts_latest_authorized_pain_score(validator):
    result = validator.validate(
        response="My pain is now 8/10.",
        current_pain=PainState(value=8),
    )

    assert result.approved is True


def test_rejects_pain_score_without_authorization(validator):
    result = validator.validate(
        response="My pain is 7/10.",
        current_pain=None,
    )

    assert result.approved is False


def test_accepts_response_without_numeric_pain_claim(validator):
    result = validator.validate(
        response="I am feeling uncomfortable.",
        current_pain=None,
    )

    assert result.approved is True


def test_accepts_response_without_numeric_claim_when_pain_is_authorized(validator):
    result = validator.validate(
        response="My leg is hurting.",
        current_pain=PainState(value=6),
    )

    assert result.approved is True


@pytest.mark.parametrize("invalid_value", [-1, 11])
def test_rejects_invalid_pain_values(invalid_value):
    with pytest.raises(ValueError):
        PainState(value=invalid_value)
