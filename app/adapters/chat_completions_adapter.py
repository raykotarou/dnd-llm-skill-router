from typing import Any

from app.adapters.canonical import CanonicalRequest


def chat_completions_to_canonical(raw_request: dict[str, Any]) -> CanonicalRequest:
    messages = raw_request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Invalid OpenAI chat completion payload: `messages` must be a list.")

    return CanonicalRequest(
        source_api="chat_completions",
        raw_request=raw_request,
        model=raw_request.get("model"),
        messages=messages,
        stream=bool(raw_request.get("stream", False)),
        temperature=raw_request.get("temperature"),
        max_tokens=raw_request.get("max_tokens"),
        top_p=raw_request.get("top_p"),
        stop=raw_request.get("stop"),
        presence_penalty=raw_request.get("presence_penalty"),
        frequency_penalty=raw_request.get("frequency_penalty"),
    )
