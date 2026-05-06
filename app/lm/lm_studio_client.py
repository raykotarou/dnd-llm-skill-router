from typing import Any, AsyncIterator

import httpx
from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from tenacity import (
    retry_if_exception,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import AppSettings


SAFE_MAIN_REQUEST_FIELDS = {
    "temperature",
    "max_tokens",
    "stop",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
}

SAFE_RESPONSES_REQUEST_FIELDS = {
    "temperature",
    "top_p",
    "max_output_tokens",
    "max_tokens",
    "previous_response_id",
    "reasoning",
    "text",
    "truncation",
    "metadata",
}

UNSUPPORTED_RESPONSES_REQUEST_FIELDS = {
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "background",
    "include",
}


def log_lm_studio_retry(retry_state) -> None:
    logger.warning(
        "LM Studio call failed, retrying attempt {}/{}: {}",
        retry_state.attempt_number,
        3,
        retry_state.outcome.exception() if retry_state.outcome else "unknown error",
    )


def should_retry_lm_studio_error(exc: BaseException) -> bool:
    if isinstance(exc, APITimeoutError):
        return False

    if isinstance(exc, httpx.TimeoutException):
        return False

    if isinstance(exc, APIConnectionError):
        return True

    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 429, 500, 502, 503, 504}

    if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)):
        return True

    return False


lm_studio_retry = retry(
    retry=retry_if_exception(should_retry_lm_studio_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    before_sleep=log_lm_studio_retry,
    reraise=True,
)


class LMStudioClient:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.lm_studio.base_url,
            api_key=settings.lm_studio.api_key,
            timeout=settings.lm_studio.request_timeout_seconds,
            max_retries=0,
        )
        self.http_client = httpx.Client(
            base_url=settings.lm_studio.base_url,
            headers={"Authorization": f"Bearer {settings.lm_studio.api_key}"},
            timeout=settings.lm_studio.request_timeout_seconds,
        )
        self.async_http_client = httpx.AsyncClient(
            base_url=settings.lm_studio.base_url,
            headers={"Authorization": f"Bearer {settings.lm_studio.api_key}"},
            timeout=(
                settings.streaming.lm_studio_timeout_seconds
                or settings.lm_studio.request_timeout_seconds
            ),
        )

    @lm_studio_retry
    def call_router_model(self, messages: list[dict]) -> dict[str, Any]:
        logger.info(
            "LM Studio router request: base_url={}, model={}, max_tokens={}, timeout={}",
            self.settings.lm_studio.base_url,
            self.settings.lm_studio.router_model,
            self.settings.generation.router_max_tokens,
            self.settings.lm_studio.request_timeout_seconds,
        )
        return self._create_chat_completion(
            {
                "model": self.settings.lm_studio.router_model,
                "messages": messages,
                "temperature": self.settings.generation.router_temperature,
                "max_tokens": self.settings.generation.router_max_tokens,
                "stream": False,
            }
        )

    @lm_studio_retry
    def call_main_model(
        self,
        messages: list[dict],
        raw_request: dict,
    ) -> dict[str, Any]:
        request_params = self._build_main_request_params(raw_request)
        logger.info(
            "LM Studio main request: base_url={}, model={}, max_tokens={}, timeout={}",
            self.settings.lm_studio.base_url,
            self.settings.lm_studio.main_model,
            request_params.get("max_tokens"),
            self.settings.lm_studio.request_timeout_seconds,
        )
        return self._create_chat_completion(
            {
                "model": self.settings.lm_studio.main_model,
                "messages": messages,
                **request_params,
            }
        )

    def _create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http_client.post("/chat/completions", json=payload)
        if response.is_error:
            logger.error(
                "LM Studio chat completion error: status={}, body={}",
                response.status_code,
                response.text,
            )
        response.raise_for_status()
        return response.json()

    def _build_main_request_params(self, raw_request: dict) -> dict[str, Any]:
        request_params: dict[str, Any] = {
            "temperature": self.settings.generation.main_temperature,
            "max_tokens": self.settings.generation.main_max_tokens,
            "stream": False,
        }

        for field in SAFE_MAIN_REQUEST_FIELDS:
            if field in raw_request:
                request_params[field] = raw_request[field]

        request_params["stream"] = False
        return request_params

    async def stream_main_model(
        self,
        messages: list[dict],
        raw_request: dict,
    ) -> AsyncIterator[str]:
        request_params = self._build_main_request_params(raw_request)
        request_params["stream"] = True  # Force stream=True for streaming

        payload = {
            "model": self.settings.lm_studio.main_model,
            "messages": messages,
            **request_params,
        }

        logger.info(
            "LM Studio streaming main request: base_url={}, model={}, max_tokens={}, timeout={}",
            self.settings.lm_studio.base_url,
            self.settings.lm_studio.main_model,
            request_params.get("max_tokens"),
            self.settings.streaming.lm_studio_timeout_seconds or self.settings.lm_studio.request_timeout_seconds,
        )

        async with self.async_http_client.stream(
            "POST",
            "/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    yield f"{line}\n\n"
                elif line == "data: [DONE]":
                    yield "data: [DONE]\n\n"
                else:
                    # Pass through other lines
                    yield f"{line}\n\n"

    @lm_studio_retry
    async def call_responses_model(
        self,
        enriched_messages: list[dict],
        raw_request: dict,
    ) -> dict[str, Any]:
        payload = self._build_responses_payload(
            enriched_messages=enriched_messages,
            raw_request=raw_request,
            stream=False,
        )
        logger.info(
            "LM Studio responses request: base_url={}, model={}, max_output_tokens={}, timeout={}",
            self.settings.lm_studio.base_url,
            self.settings.lm_studio.main_model,
            payload.get("max_output_tokens", payload.get("max_tokens")),
            self.settings.lm_studio.request_timeout_seconds,
        )
        response = await self.async_http_client.post("/responses", json=payload)
        if getattr(response, "is_error", False):
            logger.error(
                "LM Studio responses error: status={}, body={}",
                response.status_code,
                response.text,
            )
        response.raise_for_status()
        return response.json()

    async def stream_responses_model(
        self,
        enriched_messages: list[dict],
        raw_request: dict,
    ) -> AsyncIterator[bytes]:
        payload = self._build_responses_payload(
            enriched_messages=enriched_messages,
            raw_request=raw_request,
            stream=True,
        )
        logger.info(
            "LM Studio streaming responses request: base_url={}, model={}, max_output_tokens={}, timeout={}",
            self.settings.lm_studio.base_url,
            self.settings.lm_studio.main_model,
            payload.get("max_output_tokens", payload.get("max_tokens")),
            self.settings.streaming.lm_studio_timeout_seconds
            or self.settings.lm_studio.request_timeout_seconds,
        )
        async with self.async_http_client.stream(
            "POST",
            "/responses",
            json=payload,
        ) as response:
            if getattr(response, "is_error", False):
                body = await response.aread()
                logger.error(
                    "LM Studio streaming responses error: status={}, body={}",
                    response.status_code,
                    body.decode("utf-8", errors="replace"),
                )
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk

    def _build_responses_payload(
        self,
        enriched_messages: list[dict],
        raw_request: dict,
        stream: bool,
    ) -> dict[str, Any]:
        request_params: dict[str, Any] = {}

        for field in SAFE_RESPONSES_REQUEST_FIELDS:
            if field in raw_request:
                request_params[field] = raw_request[field]

        if "max_tokens" in request_params and "max_output_tokens" not in request_params:
            request_params["max_output_tokens"] = request_params.pop("max_tokens")

        unsupported_fields = [
            field for field in sorted(UNSUPPORTED_RESPONSES_REQUEST_FIELDS) if field in raw_request
        ]
        if unsupported_fields:
            message = (
                "Responses request contains unsupported fields for MVP: "
                f"{unsupported_fields}"
            )
            if self.settings.responses_api.unsupported_tools_policy == "reject":
                raise ValueError(message)
            logger.warning("{}; ignoring them", message)

        return {
            **request_params,
            "model": self.settings.lm_studio.main_model,
            "input": responses_input_from_messages(enriched_messages),
            "stream": stream,
        }

    def close(self) -> None:
        self.http_client.close()

    async def aclose(self) -> None:
        await self.async_http_client.aclose()


def responses_input_from_messages(messages: list[dict]) -> list[dict[str, Any]]:
    responses_input = []
    for message in messages:
        role = message.get("role", "user")
        if role == "developer":
            role = "system"

        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content_type = "output_text" if role == "assistant" else "input_text"

        responses_input.append(
            {
                "role": role,
                "content": [
                    {
                        "type": content_type,
                        "text": content,
                    }
                ],
            }
        )

    return responses_input
