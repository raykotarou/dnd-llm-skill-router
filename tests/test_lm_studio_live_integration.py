import asyncio
import os

import httpx
import pytest

from app.config.settings import load_settings
from app.lm.lm_studio_client import LMStudioClient
from app.routing.skill_ranker import build_router_messages, parse_router_response


pytestmark = [
    pytest.mark.lm_studio_live,
    pytest.mark.skipif(
        os.getenv("RUN_LM_STUDIO_INTEGRATION") != "1",
        reason=(
            "Live LM Studio integration tests are disabled. "
            "Set RUN_LM_STUDIO_INTEGRATION=1 to run them."
        ),
    ),
]


def _timeout() -> float:
    return float(os.getenv("LM_STUDIO_LIVE_TIMEOUT", "900"))


def _max_output_tokens() -> int:
    return int(os.getenv("LM_STUDIO_LIVE_MAX_OUTPUT_TOKENS", "80"))


def _live_settings():
    settings = load_settings()
    lm_studio = settings.lm_studio.model_copy(
        update={"request_timeout_seconds": _timeout()}
    )
    streaming = settings.streaming.model_copy(
        update={"lm_studio_timeout_seconds": _timeout()}
    )
    return settings.model_copy(
        update={
            "lm_studio": lm_studio,
            "streaming": streaming,
        }
    )


def test_lm_studio_models_endpoint_is_reachable() -> None:
    settings = _live_settings()

    with httpx.Client(timeout=30) as client:
        response = client.get(f"{settings.lm_studio.base_url}/models")

    assert response.status_code == 200
    body = response.json()
    model_ids = {model["id"] for model in body.get("data", [])}
    assert settings.lm_studio.router_model in model_ids
    assert settings.lm_studio.main_model in model_ids


def test_live_router_model_accepts_bounded_router_prompt() -> None:
    settings = _live_settings()
    client = LMStudioClient(settings)

    try:
        router_messages = build_router_messages(
            messages=[
                {"role": "user", "content": "Придумай сцену в таверне"},
                {"role": "assistant", "content": "Короткий ответ"},
                {
                    "role": "user",
                    "content": "Помоги написать историю на основе контекста",
                },
            ],
            latest_user_message="Помоги написать историю на основе контекста",
        )

        response = client.call_router_model(router_messages)
    finally:
        client.close()
        asyncio.run(client.aclose())

    parsed = parse_router_response(response, max_ranked_skills=3)
    assert parsed.ranked_skills
    assert parsed.selected_skill in {"story", "analysis", "template", "lore", "rules"}
    assert parsed.confidence is not None


def test_live_responses_non_stream_accepts_assistant_history() -> None:
    settings = _live_settings()
    client = LMStudioClient(settings)

    async def run_test() -> dict:
        try:
            return await client.call_responses_model(
                enriched_messages=[
                    {"role": "system", "content": "Отвечай одним коротким предложением."},
                    {"role": "user", "content": "Скажи: первый ход."},
                    {"role": "assistant", "content": "Первый ход."},
                    {"role": "user", "content": "Скажи: второй ход."},
                ],
                raw_request={
                    "stream": False,
                    "temperature": 0.0,
                    "max_output_tokens": _max_output_tokens(),
                    "text": {"format": {"type": "text"}},
                },
            )
        finally:
            client.close()
            await client.aclose()

    response = asyncio.run(asyncio.wait_for(run_test(), timeout=_timeout()))

    assert response.get("object") == "response"
    assert response.get("status") in {"completed", "incomplete"}
    assert response.get("output")


def test_live_responses_stream_yields_sse_with_assistant_history() -> None:
    settings = _live_settings()
    client = LMStudioClient(settings)

    async def collect_first_chunks() -> list[bytes]:
        chunks = []
        try:
            async for chunk in client.stream_responses_model(
                enriched_messages=[
                    {"role": "system", "content": "Отвечай коротко."},
                    {"role": "user", "content": "Скажи: первый ход."},
                    {"role": "assistant", "content": "Первый ход."},
                    {"role": "user", "content": "Скажи: второй ход."},
                ],
                raw_request={
                    "stream": True,
                    "temperature": 0.0,
                    "max_output_tokens": _max_output_tokens(),
                    "text": {"format": {"type": "text"}},
                },
            ):
                chunks.append(chunk)
                if b"response.output_text.delta" in b"".join(chunks):
                    break
            return chunks
        finally:
            client.close()
            await client.aclose()

    chunks = asyncio.run(asyncio.wait_for(collect_first_chunks(), timeout=_timeout()))
    joined = b"".join(chunks)

    assert b"event:" in joined or b"data:" in joined
    assert b"response.created" in joined or b"response.output_text.delta" in joined
