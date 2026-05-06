from typing import Any

from pydantic import BaseModel


class PreparedGeneration(BaseModel):
    raw_request: dict[str, Any]
    model: str
    stream: bool

    messages: list[dict[str, Any]]
    enriched_messages: list[dict[str, Any]]

    selected_skill: str | None
    confidence: int | None
    needs_clarification: bool
    clarification_message: str | None

    routing_metadata: dict[str, Any] = {}