from typing import Any, Literal

from pydantic import BaseModel, Field


class CanonicalRequest(BaseModel):
    source_api: Literal["chat_completions", "responses"]
    raw_request: dict[str, Any]

    model: str | None = None
    messages: list[dict[str, Any]]

    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    previous_response_id: str | None = None
    unsupported_fields: list[str] = Field(default_factory=list)

    def to_chat_payload(self) -> dict[str, Any]:
        return {
            **self.raw_request,
            "_source_api": self.source_api,
            "_unsupported_fields": self.unsupported_fields,
            "model": self.model or self.raw_request.get("model", "dnd-skill-router"),
            "messages": self.messages,
            "stream": self.stream,
        }
