from typing import Any

from app.adapters.canonical import CanonicalRequest


TEXT_PART_TYPES = {"input_text", "output_text", "text"}
ROLE_MAP = {
    "developer": "system",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}
UNSUPPORTED_RESPONSE_FIELDS = {
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "background",
    "include",
}


def responses_to_canonical(raw_request: dict[str, Any]) -> CanonicalRequest:
    if "input" not in raw_request:
        raise ValueError("Invalid Responses payload: `input` is required.")

    messages = _normalize_input(raw_request["input"])
    unsupported_fields = [
        field for field in sorted(UNSUPPORTED_RESPONSE_FIELDS) if field in raw_request
    ]

    return CanonicalRequest(
        source_api="responses",
        raw_request=raw_request,
        model=raw_request.get("model"),
        messages=messages,
        stream=bool(raw_request.get("stream", False)),
        temperature=raw_request.get("temperature"),
        max_tokens=raw_request.get("max_output_tokens", raw_request.get("max_tokens")),
        top_p=raw_request.get("top_p"),
        stop=raw_request.get("stop"),
        presence_penalty=raw_request.get("presence_penalty"),
        frequency_penalty=raw_request.get("frequency_penalty"),
        previous_response_id=raw_request.get("previous_response_id"),
        unsupported_fields=unsupported_fields,
    )


def _normalize_input(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if not isinstance(input_value, list):
        raise ValueError("Invalid Responses payload: `input` must be a string or a list.")

    messages = []
    for item in input_value:
        if not isinstance(item, dict):
            raise ValueError("Invalid Responses payload: each `input` item must be an object.")

        role = _normalize_role(item.get("role", "user"))
        content = _normalize_content(item.get("content", ""))
        messages.append({"role": role, "content": content})

    return messages


def _normalize_role(role: Any) -> str:
    if not isinstance(role, str) or role not in ROLE_MAP:
        raise ValueError(f"Invalid Responses payload: unsupported role `{role}`.")

    return ROLE_MAP[role]


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        raise ValueError(
            "Invalid Responses payload: message `content` must be a string or a list."
        )

    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Invalid Responses payload: content parts must be objects.")

        part_type = part.get("type")
        if part_type not in TEXT_PART_TYPES:
            raise ValueError(
                f"Invalid Responses payload: unsupported content part type `{part_type}`."
            )

        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Invalid Responses payload: text content part requires `text`.")

        parts.append(text)

    return "\n".join(parts)
