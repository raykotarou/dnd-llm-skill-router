import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.openai_proxy as openai_proxy
from app.graph.prepared import PreparedGeneration
from app.lm.lm_studio_client import LMStudioClient
from app.config.settings import load_settings


def _prepared(stream: bool) -> PreparedGeneration:
    return PreparedGeneration(
        raw_request={
            "model": "dnd-skill-router",
            "messages": [{"role": "user", "content": "Придумай сцену"}],
            "stream": stream,
        },
        model="dnd-skill-router",
        stream=stream,
        messages=[{"role": "user", "content": "Придумай сцену"}],
        enriched_messages=[{"role": "user", "content": "Придумай сцену"}],
        selected_skill="story",
        confidence=91,
        needs_clarification=False,
        clarification_message=None,
    )


def _test_client() -> TestClient:
    app = FastAPI()
    app.include_router(openai_proxy.router)
    return TestClient(app)


def test_chat_completions_returns_streaming_response_for_stream_requests(monkeypatch) -> None:
    async def fake_prepare(payload: dict) -> PreparedGeneration:
        return _prepared(stream=True)

    async def fake_stream(prepared: PreparedGeneration):
        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":"Готово"}}]}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_complete(prepared: PreparedGeneration) -> dict:
        raise AssertionError("complete_generation must not run for stream=true")

    monkeypatch.setattr(openai_proxy, "prepare_generation", fake_prepare)
    monkeypatch.setattr(openai_proxy, "stream_generation", fake_stream)
    monkeypatch.setattr(openai_proxy, "complete_generation", fake_complete)

    response = _test_client().post(
        "/v1/chat/completions",
        json={
            "model": "dnd-skill-router",
            "messages": [{"role": "user", "content": "Придумай сцену"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Готово" in response.text
    assert "data: [DONE]" in response.text


def test_chat_completions_returns_json_for_non_stream_requests(monkeypatch) -> None:
    async def fake_prepare(payload: dict) -> PreparedGeneration:
        return _prepared(stream=False)

    async def fake_stream(prepared: PreparedGeneration):
        raise AssertionError("stream_generation must not run for stream=false")
        yield ""

    async def fake_complete(prepared: PreparedGeneration) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "dnd-skill-router",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Готово"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(openai_proxy, "prepare_generation", fake_prepare)
    monkeypatch.setattr(openai_proxy, "stream_generation", fake_stream)
    monkeypatch.setattr(openai_proxy, "complete_generation", fake_complete)

    response = _test_client().post(
        "/v1/chat/completions",
        json={
            "model": "dnd-skill-router",
            "messages": [{"role": "user", "content": "Придумай сцену"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["choices"][0]["message"]["content"] == "Готово"


def test_lm_studio_stream_main_model_proxies_sse_lines() -> None:
    settings = load_settings()
    client = LMStudioClient(settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"А"}}]}'
            yield ""
            yield 'data: {"choices":[{"delta":{"content":"Б"}}]}'
            yield "data: [DONE]"

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.payload = None

        def stream(self, method: str, url: str, json: dict):
            self.payload = json
            return FakeStreamContext()

        async def aclose(self) -> None:
            return None

    fake_async_client = FakeAsyncClient()
    client.async_http_client = fake_async_client

    async def collect_chunks() -> list[str]:
        return [
            chunk
            async for chunk in client.stream_main_model(
                messages=[{"role": "user", "content": "Привет"}],
                raw_request={"stream": True, "max_tokens": 7},
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert fake_async_client.payload["stream"] is True
    assert fake_async_client.payload["max_tokens"] == 7
    assert chunks == [
        'data: {"choices":[{"delta":{"content":"А"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"Б"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
