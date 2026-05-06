import pytest
from pydantic import ValidationError

from app.config.settings import load_settings


def test_responses_reasoning_settings_are_loaded() -> None:
    settings = load_settings()

    assert settings.responses_api.reasoning.mode in {
        "drop",
        "think_block",
        "plain",
        "pass_through",
    }
    assert (
        settings.responses_api.reasoning.stream_insertion_strategy
        == "transform_reasoning_events"
    )


def test_invalid_reasoning_mode_is_rejected() -> None:
    settings = load_settings()
    payload = settings.model_dump()
    payload["responses_api"]["reasoning"]["mode"] = "invalid"

    with pytest.raises(ValidationError):
        type(settings).model_validate(payload)


def test_invalid_diagnostics_placement_is_rejected() -> None:
    settings = load_settings()
    payload = settings.model_dump()
    payload["responses_api"]["diagnostics"]["placement"] = "middle"

    with pytest.raises(ValidationError):
        type(settings).model_validate(payload)


def test_invalid_diagnostics_format_is_rejected() -> None:
    settings = load_settings()
    payload = settings.model_dump()
    payload["responses_api"]["diagnostics"]["format"] = "json"

    with pytest.raises(ValidationError):
        type(settings).model_validate(payload)
