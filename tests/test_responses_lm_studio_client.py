import asyncio

from app.config.settings import load_settings
from app.lm.lm_studio_client import LMStudioClient, responses_input_from_messages


def test_responses_input_from_messages_converts_enriched_messages() -> None:
    result = responses_input_from_messages(
        [
            {"role": "system", "content": "Base prompt"},
            {"role": "developer", "content": "Developer prompt"},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Предыдущий ответ"},
        ]
    )

    assert result == [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "Base prompt"}],
        },
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "Developer prompt"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Привет"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Предыдущий ответ"}],
        },
    ]


def test_responses_input_uses_output_text_for_assistant_history() -> None:
    result = responses_input_from_messages(
        [
            {"role": "user", "content": "Первый запрос"},
            {"role": "assistant", "content": "<think>...</think>\nОтвет"},
            {"role": "user", "content": "Второй запрос"},
        ]
    )

    assert result[0]["content"][0]["type"] == "input_text"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"][0]["type"] == "output_text"
    assert result[2]["content"][0]["type"] == "input_text"


def test_call_responses_model_posts_to_responses_endpoint() -> None:
    settings = load_settings()
    client = LMStudioClient(settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"id": "resp_test", "object": "response"}

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, url: str, json: dict):
            self.calls.append((url, json))
            return FakeResponse()

        async def aclose(self) -> None:
            return None

    fake_async_client = FakeAsyncClient()
    client.async_http_client = fake_async_client

    response = asyncio.run(
        client.call_responses_model(
            enriched_messages=[{"role": "user", "content": "Привет"}],
            raw_request={
                "model": "dnd-skill-router",
                "max_output_tokens": 42,
                "previous_response_id": "resp_prev",
                "tools": [{"type": "mcp"}],
            },
        )
    )

    url, payload = fake_async_client.calls[0]
    assert response["id"] == "resp_test"
    assert url == "/responses"
    assert payload["model"] == settings.lm_studio.main_model
    assert payload["stream"] is False
    assert payload["max_output_tokens"] == 42
    assert payload["previous_response_id"] == "resp_prev"
    assert "tools" not in payload


def test_stream_responses_model_uses_byte_level_passthrough() -> None:
    settings = load_settings()
    client = LMStudioClient(settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"event: response.created\n"
            yield b"data: {}\n\n"

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.payload = None
            self.url = None

        def stream(self, method: str, url: str, json: dict):
            self.url = url
            self.payload = json
            return FakeStreamContext()

        async def aclose(self) -> None:
            return None

    fake_async_client = FakeAsyncClient()
    client.async_http_client = fake_async_client

    async def collect() -> list[bytes]:
        return [
            chunk
            async for chunk in client.stream_responses_model(
                enriched_messages=[{"role": "user", "content": "Привет"}],
                raw_request={"stream": True},
            )
        ]

    chunks = asyncio.run(collect())

    assert fake_async_client.url == "/responses"
    assert fake_async_client.payload["stream"] is True
    assert chunks == [b"event: response.created\n", b"data: {}\n\n"]
