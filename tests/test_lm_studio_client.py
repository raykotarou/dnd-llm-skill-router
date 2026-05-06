from openai import APIConnectionError, APITimeoutError

from app.config.settings import load_settings
from app.lm.lm_studio_client import LMStudioClient, should_retry_lm_studio_error


def test_lm_studio_client_uses_configured_timeout() -> None:
    settings = load_settings()
    client = LMStudioClient(settings)

    assert client.client.timeout == settings.lm_studio.request_timeout_seconds
    assert client.client.max_retries == 0


def test_lm_studio_retry_policy_does_not_retry_generation_timeout() -> None:
    assert should_retry_lm_studio_error(APITimeoutError(request=None)) is False


def test_lm_studio_retry_policy_retries_connection_errors() -> None:
    assert should_retry_lm_studio_error(APIConnectionError(request=None)) is True
