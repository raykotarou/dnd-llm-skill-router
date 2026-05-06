from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.openai_proxy as openai_proxy
from app.config.settings import load_settings


class FakeLMClient:
    def __init__(
        self,
        router_confidence: int = 91,
        router_skill: str = "story",
        response_payload: dict | None = None,
        stream_chunks: list[bytes] | None = None,
    ) -> None:
        self.router_confidence = router_confidence
        self.router_skill = router_skill
        self.response_payload = response_payload
        self.stream_chunks = stream_chunks
        self.router_calls = 0
        self.responses_calls = 0
        self.stream_responses_calls = 0
        self.last_raw_request = None

    def call_router_model(self, messages: list[dict]) -> dict:
        self.router_calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"ranked_skills":[{"skill":"'
                            f'{self.router_skill}","confidence":{self.router_confidence},'
                            '"reason":"test"}]}'
                        )
                    }
                }
            ]
        }

    async def call_responses_model(
        self,
        enriched_messages: list[dict],
        raw_request: dict,
    ) -> dict:
        self.responses_calls += 1
        self.last_raw_request = raw_request
        if self.response_payload is not None:
            return self.response_payload
        return {
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "model": "qwen-main",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Готово"}],
                }
            ],
        }

    async def stream_responses_model(
        self,
        enriched_messages: list[dict],
        raw_request: dict,
    ):
        self.stream_responses_calls += 1
        self.last_raw_request = raw_request
        if self.stream_chunks is not None:
            for chunk in self.stream_chunks:
                yield chunk
            return
        yield b"event: response.created\ndata: {}\n\n"
        yield 'event: response.output_text.delta\ndata: {"delta":"Г"}\n\n'.encode(
            "utf-8"
        )
        yield b"event: response.completed\ndata: {}\n\n"

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _settings_with_reasoning(mode: str):
    settings = load_settings()
    reasoning = settings.responses_api.reasoning.model_copy(update={"mode": mode})
    responses_api = settings.responses_api.model_copy(update={"reasoning": reasoning})
    return settings.model_copy(update={"responses_api": responses_api})


def _client(fake_lm: FakeLMClient, monkeypatch, settings=None) -> TestClient:
    if settings is not None:
        monkeypatch.setattr(openai_proxy, "load_settings", lambda: settings)
        monkeypatch.setattr("app.graph.pipeline.load_settings", lambda: settings)
    monkeypatch.setattr("app.graph.pipeline.LMStudioClient", lambda settings: fake_lm)
    app = FastAPI()
    app.include_router(openai_proxy.router)
    return TestClient(app)


def test_responses_endpoint_returns_response_json(monkeypatch) -> None:
    fake_lm = FakeLMClient()
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "dnd-skill-router",
            "input": "Придумай сцену в таверне",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["object"] == "response"
    assert fake_lm.router_calls == 1
    assert fake_lm.responses_calls == 1


def test_responses_endpoint_returns_event_stream(monkeypatch) -> None:
    fake_lm = FakeLMClient()
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "dnd-skill-router",
            "input": "Придумай сцену в таверне",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: response.output_text.delta" in response.text
    assert fake_lm.stream_responses_calls == 1


def test_manual_command_works_through_responses_input(monkeypatch) -> None:
    fake_lm = FakeLMClient()
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "dnd-skill-router",
            "input": "!analysis Проверь сцену на несостыковки",
        },
    )

    assert response.status_code == 200
    assert fake_lm.router_calls == 0
    assert fake_lm.responses_calls == 1


def test_low_confidence_responses_returns_clarification_json(monkeypatch) -> None:
    fake_lm = FakeLMClient(router_confidence=79)
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "dnd-skill-router",
            "input": "Помоги с этим",
            "stream": False,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["id"] == "resp_local_clarification"
    assert body["object"] == "response"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert "Я не уверен" in body["output"][0]["content"][0]["text"]
    assert fake_lm.responses_calls == 0


def test_low_confidence_responses_returns_clarification_stream(monkeypatch) -> None:
    fake_lm = FakeLMClient(router_confidence=79)
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "dnd-skill-router",
            "input": "Помоги с этим",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: response.created" in response.text
    assert "event: response.output_text.delta" in response.text
    assert "event: response.completed" in response.text
    assert fake_lm.stream_responses_calls == 0


def test_previous_response_id_is_passed_to_lm_studio_payload(monkeypatch) -> None:
    fake_lm = FakeLMClient()
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={
            "model": "dnd-skill-router",
            "input": "Продолжи",
            "previous_response_id": "resp_prev",
        },
    )

    assert response.status_code == 200
    assert fake_lm.last_raw_request["previous_response_id"] == "resp_prev"


def test_unknown_responses_role_returns_400(monkeypatch) -> None:
    fake_lm = FakeLMClient()
    client = _client(fake_lm, monkeypatch)

    response = client.post(
        "/v1/responses",
        json={"input": [{"role": "unknown", "content": "Привет"}]},
    )

    assert response.status_code == 400
    assert "unsupported role" in response.json()["detail"]


def _reasoning_response_payload() -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "model": "qwen-main",
        "output": [
            {
                "id": "rsn",
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "думал"}],
            },
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ответ"}],
            },
        ],
        "usage": {"output_tokens_details": {"reasoning_tokens": 5}},
    }


def _reasoning_stream_chunks() -> list[bytes]:
    return [
        b'event: response.created\ndata: {"type":"response.created"}\n\n',
        b'event: response.output_item.added\ndata: {"item":{"type":"reasoning"}}\n\n',
        b'event: response.reasoning_text.delta\ndata: {"type":"response.reasoning_text.delta","delta":"think"}\n\n',
        b'event: response.reasoning_text.done\ndata: {"type":"response.reasoning_text.done"}\n\n',
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"answer"}\n\n',
        (
            b'event: response.completed\ndata: {"type":"response.completed",'
            b'"response":{"output":[{"type":"reasoning"},{"type":"message"}],'
            b'"usage":{"output_tokens_details":{"reasoning_tokens":5}}}}\n\n'
        ),
    ]


def test_stream_drop_removes_reasoning_from_downstream(monkeypatch) -> None:
    fake_lm = FakeLMClient(stream_chunks=_reasoning_stream_chunks())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("drop"))

    response = client.post("/v1/responses", json={"input": "Привет", "stream": True})

    assert "response.reasoning_text.delta" not in response.text
    assert '"delta":"think"' not in response.text
    assert "response.output_text.delta" in response.text
    assert "answer" in response.text


def test_stream_think_block_transforms_reasoning_to_think_output(monkeypatch) -> None:
    fake_lm = FakeLMClient(stream_chunks=_reasoning_stream_chunks())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("think_block"))

    response = client.post("/v1/responses", json={"input": "Привет", "stream": True})

    assert "response.reasoning_text.delta" not in response.text
    assert "<think>\\nthink" in response.text
    assert "\\n</think>\\n\\n" in response.text


def test_stream_plain_transforms_reasoning_without_tags(monkeypatch) -> None:
    fake_lm = FakeLMClient(stream_chunks=_reasoning_stream_chunks())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("plain"))

    response = client.post("/v1/responses", json={"input": "Привет", "stream": True})

    assert "response.reasoning_text.delta" not in response.text
    assert '"delta": "think"' in response.text
    assert "<think>" not in response.text


def test_stream_pass_through_keeps_raw_reasoning(monkeypatch) -> None:
    fake_lm = FakeLMClient(stream_chunks=_reasoning_stream_chunks())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("pass_through"))

    response = client.post("/v1/responses", json={"input": "Привет", "stream": True})

    assert "response.reasoning_text.delta" in response.text
    assert '"delta": "think"' in response.text


def test_non_stream_drop_removes_reasoning_from_json(monkeypatch) -> None:
    fake_lm = FakeLMClient(response_payload=_reasoning_response_payload())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("drop"))

    response = client.post("/v1/responses", json={"input": "Привет"})

    assert [item["type"] for item in response.json()["output"]] == ["message"]


def test_non_stream_think_block_inserts_reasoning(monkeypatch) -> None:
    fake_lm = FakeLMClient(response_payload=_reasoning_response_payload())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("think_block"))

    response = client.post("/v1/responses", json={"input": "Привет"})

    text = response.json()["output"][0]["content"][0]["text"]
    assert "<think>\nдумал\n</think>" in text


def test_non_stream_plain_inserts_reasoning_without_tags(monkeypatch) -> None:
    fake_lm = FakeLMClient(response_payload=_reasoning_response_payload())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("plain"))

    response = client.post("/v1/responses", json={"input": "Привет"})

    text = response.json()["output"][0]["content"][0]["text"]
    assert text == "думал\n\n"


def test_non_stream_pass_through_keeps_reasoning(monkeypatch) -> None:
    fake_lm = FakeLMClient(response_payload=_reasoning_response_payload())
    client = _client(fake_lm, monkeypatch, _settings_with_reasoning("pass_through"))

    response = client.post("/v1/responses", json={"input": "Привет"})

    assert [item["type"] for item in response.json()["output"]] == [
        "reasoning",
        "message",
    ]
