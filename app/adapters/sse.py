from collections.abc import AsyncIterator
import json
from typing import Any

from pydantic import BaseModel, Field


class SSEEvent(BaseModel):
    event: str | None = None
    data: dict[str, Any] | str | None = None
    raw_lines: list[str] = Field(default_factory=list)


def parse_sse_event(block: str) -> SSEEvent:
    lines = [line for line in block.splitlines() if line]
    event_type = None
    data_lines = []

    for line in lines:
        if line.startswith("event:"):
            event_type = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    data_text = "\n".join(data_lines) if data_lines else None
    parsed_data: dict[str, Any] | str | None = data_text
    if data_text:
        try:
            loaded = json.loads(data_text)
            parsed_data = loaded if isinstance(loaded, dict) else data_text
        except json.JSONDecodeError:
            parsed_data = data_text

    if event_type is None and isinstance(parsed_data, dict):
        event_type = parsed_data.get("type")

    return SSEEvent(event=event_type, data=parsed_data, raw_lines=lines)


def format_sse_event(event_type: str | None, data: dict[str, Any] | str) -> bytes:
    lines = []
    if event_type:
        lines.append(f"event: {event_type}")

    if isinstance(data, str):
        data_text = data
    else:
        data_text = json.dumps(data, ensure_ascii=False)

    for line in data_text.splitlines() or [""]:
        lines.append(f"data: {line}")

    return ("\n".join(lines) + "\n\n").encode("utf-8")


async def iter_sse_events_from_bytes(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[SSEEvent]:
    buffer = ""
    async for chunk in chunks:
        buffer += chunk.decode("utf-8", errors="replace")

        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            if block.strip():
                yield parse_sse_event(block)

    if buffer.strip():
        yield parse_sse_event(buffer)
