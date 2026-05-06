from __future__ import annotations

from copy import deepcopy
from typing import Any

from loguru import logger

from app.adapters.sse import SSEEvent


SYNTHETIC_REASONING_ITEM_ID = "msg_local_reasoning"
SYNTHETIC_DIAGNOSTICS_ITEM_ID = "msg_local_diagnostics"
SYNTHETIC_REASONING_OUTPUT_INDEX = 999_999
SYNTHETIC_DIAGNOSTICS_OUTPUT_INDEX = 999_998


class ResponsesReasoningStreamTransformer:
    def __init__(
        self,
        reasoning_mode: str,
        stream_insertion_strategy: str,
        diagnostics_config: dict[str, Any],
        routing_metadata: dict[str, Any],
        strip_reasoning_from_completed: bool = True,
        log_presence: bool = True,
        log_raw_reasoning: bool = False,
    ) -> None:
        self.reasoning_mode = reasoning_mode
        self.stream_insertion_strategy = stream_insertion_strategy
        self.diagnostics_config = diagnostics_config
        self.routing_metadata = routing_metadata
        self.strip_reasoning_from_completed = strip_reasoning_from_completed
        self.log_presence = log_presence
        self.log_raw_reasoning = log_raw_reasoning

        self.reasoning_present = False
        self.reasoning_text_parts: list[str] = []
        self.synthetic_reasoning_started = False
        self.synthetic_reasoning_done = False
        self.start_diagnostics_sent = False

    def transform(self, event: SSEEvent) -> list[SSEEvent]:
        if event.event == "response.created":
            return [event, *self._start_diagnostics_events_if_needed()]

        if self.reasoning_mode == "pass_through":
            return self._with_start_diagnostics([event])

        event_type = event.event
        data = event.data
        output_events: list[SSEEvent] = []

        output_events.extend(self._start_diagnostics_events_if_needed())

        if self._is_reasoning_event(event):
            self.reasoning_present = True
            if self._is_reasoning_delta(event):
                delta = self._extract_delta(data)
                if delta:
                    self.reasoning_text_parts.append(delta)
                    if self.log_raw_reasoning:
                        logger.debug("Responses reasoning delta: {}", delta)
                if self.reasoning_mode in {"think_block", "plain"}:
                    output_events.extend(self._synthetic_reasoning_delta_events(delta))
            elif self._is_reasoning_done(event):
                if self.reasoning_mode in {"think_block", "plain"}:
                    output_events.extend(self._synthetic_reasoning_done_events())
            return output_events

        if event_type == "response.completed" and isinstance(data, dict):
            output_events.extend(self._end_diagnostics_events_if_needed(data))
            transformed = transform_completed_response(
                completed_event_data=data,
                reasoning_mode=self.reasoning_mode,
                synthetic_reasoning_message=self._synthetic_reasoning_message_or_none(),
                strip_reasoning=self.strip_reasoning_from_completed,
            )
            output_events.append(SSEEvent(event=event_type, data=transformed))
            self._log_reasoning_metadata(transformed)
            return output_events

        output_events.append(event)
        return output_events

    def _with_start_diagnostics(self, events: list[SSEEvent]) -> list[SSEEvent]:
        return [*self._start_diagnostics_events_if_needed(), *events]

    def _start_diagnostics_events_if_needed(self) -> list[SSEEvent]:
        if self.start_diagnostics_sent:
            return []
        if not self.diagnostics_config.get("enabled"):
            return []
        if self.diagnostics_config.get("placement") not in {"start", "both"}:
            return []

        self.start_diagnostics_sent = True
        return build_synthetic_message_stream_events(
            item_id=SYNTHETIC_DIAGNOSTICS_ITEM_ID,
            output_index=SYNTHETIC_DIAGNOSTICS_OUTPUT_INDEX,
            text=build_diagnostics_text(
                source_api="responses",
                reasoning_mode=self.reasoning_mode,
                stream_insertion_strategy=self.stream_insertion_strategy,
                selected_skill=self.routing_metadata.get("selected_skill"),
                confidence=self.routing_metadata.get("confidence"),
                manual_skill=self.routing_metadata.get("manual_skill"),
                format=self.diagnostics_config.get("format", "visible_block"),
            ),
        )

    def _end_diagnostics_events_if_needed(self, completed_data: dict[str, Any]) -> list[SSEEvent]:
        if not self.diagnostics_config.get("enabled"):
            return []
        if self.diagnostics_config.get("placement") not in {"end", "both"}:
            return []

        reasoning_tokens = _extract_reasoning_tokens(completed_data)
        return build_synthetic_message_stream_events(
            item_id=f"{SYNTHETIC_DIAGNOSTICS_ITEM_ID}_end",
            output_index=SYNTHETIC_DIAGNOSTICS_OUTPUT_INDEX - 1,
            text=build_diagnostics_text(
                source_api="responses",
                reasoning_mode=self.reasoning_mode,
                stream_insertion_strategy=self.stream_insertion_strategy,
                selected_skill=self.routing_metadata.get("selected_skill"),
                confidence=self.routing_metadata.get("confidence"),
                manual_skill=self.routing_metadata.get("manual_skill"),
                format=self.diagnostics_config.get("format", "visible_block"),
                reasoning_present=self.reasoning_present,
                reasoning_tokens=reasoning_tokens,
            ),
        )

    def _is_reasoning_event(self, event: SSEEvent) -> bool:
        data = event.data if isinstance(event.data, dict) else {}
        event_type = event.event

        if event_type in {"response.reasoning_text.delta", "response.reasoning_text.done"}:
            return True
        if event_type == "response.output_item.added":
            return _nested_type(data, "item") == "reasoning"
        if event_type == "response.output_item.done":
            return _nested_type(data, "item") == "reasoning"
        if event_type == "response.content_part.added":
            return _nested_type(data, "part") == "reasoning_text"
        if event_type == "response.content_part.done":
            return _nested_type(data, "part") == "reasoning_text"
        return False

    def _is_reasoning_delta(self, event: SSEEvent) -> bool:
        return event.event == "response.reasoning_text.delta"

    def _is_reasoning_done(self, event: SSEEvent) -> bool:
        return event.event == "response.reasoning_text.done"

    def _extract_delta(self, data: dict[str, Any] | str | None) -> str:
        if not isinstance(data, dict):
            return ""
        delta = data.get("delta", "")
        return delta if isinstance(delta, str) else str(delta)

    def _synthetic_reasoning_delta_events(self, delta: str) -> list[SSEEvent]:
        events = []
        if not self.synthetic_reasoning_started:
            self.synthetic_reasoning_started = True
            events.append(
                build_synthetic_output_item_added(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                )
            )
            events.append(
                build_synthetic_content_part_added(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    content_index=0,
                )
            )
            if self.reasoning_mode == "think_block":
                delta = f"<think>\n{delta}"

        if delta:
            events.append(
                build_synthetic_output_text_delta(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    content_index=0,
                    delta=delta,
                )
            )
        return events

    def _synthetic_reasoning_done_events(self) -> list[SSEEvent]:
        if not self.synthetic_reasoning_started or self.synthetic_reasoning_done:
            return []

        self.synthetic_reasoning_done = True
        events = []
        if self.reasoning_mode == "think_block":
            events.append(
                build_synthetic_output_text_delta(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    content_index=0,
                    delta="\n</think>\n\n",
                )
            )
        elif self.reasoning_mode == "plain":
            events.append(
                build_synthetic_output_text_delta(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    content_index=0,
                    delta="\n\n",
                )
            )

        events.extend(
            [
                build_synthetic_output_text_done(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    content_index=0,
                    text=self._synthetic_reasoning_text(),
                ),
                build_synthetic_content_part_done(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    content_index=0,
                    text=self._synthetic_reasoning_text(),
                ),
                build_synthetic_output_item_done(
                    item_id=SYNTHETIC_REASONING_ITEM_ID,
                    output_index=SYNTHETIC_REASONING_OUTPUT_INDEX,
                    text=self._synthetic_reasoning_text(),
                ),
            ]
        )
        return events

    def _synthetic_reasoning_text(self) -> str:
        reasoning_text = "".join(self.reasoning_text_parts)
        if not reasoning_text:
            return ""
        if self.reasoning_mode == "think_block":
            return f"<think>\n{reasoning_text}\n</think>\n\n"
        if self.reasoning_mode == "plain":
            return f"{reasoning_text}\n\n"
        return reasoning_text

    def _synthetic_reasoning_message_or_none(self) -> dict[str, Any] | None:
        if not self.reasoning_present:
            return None
        text = self._synthetic_reasoning_text()
        if not text:
            return None
        return build_synthetic_message_item(SYNTHETIC_REASONING_ITEM_ID, text)

    def _log_reasoning_metadata(self, completed_data: dict[str, Any]) -> None:
        if not self.log_presence:
            return

        logger.info(
            "{}",
            {
                "source_api": "responses",
                "reasoning_mode": self.reasoning_mode,
                "stream_insertion_strategy": self.stream_insertion_strategy,
                "reasoning_present": self.reasoning_present,
                "reasoning_tokens": _extract_reasoning_tokens(completed_data),
                "selected_skill": self.routing_metadata.get("selected_skill"),
                "confidence": self.routing_metadata.get("confidence"),
                "manual_skill": self.routing_metadata.get("manual_skill"),
            },
        )


def build_synthetic_output_item_added(item_id: str, output_index: int) -> SSEEvent:
    data = {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    }
    return SSEEvent(event="response.output_item.added", data=data)


def build_synthetic_content_part_added(
    item_id: str,
    output_index: int,
    content_index: int,
) -> SSEEvent:
    data = {
        "type": "response.content_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
    }
    return SSEEvent(event="response.content_part.added", data=data)


def build_synthetic_output_text_delta(
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> SSEEvent:
    data = {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta,
    }
    return SSEEvent(event="response.output_text.delta", data=data)


def build_synthetic_output_text_done(
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> SSEEvent:
    data = {
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "text": text,
    }
    return SSEEvent(event="response.output_text.done", data=data)


def build_synthetic_content_part_done(
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> SSEEvent:
    data = {
        "type": "response.content_part.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": {
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        },
    }
    return SSEEvent(event="response.content_part.done", data=data)


def build_synthetic_output_item_done(
    item_id: str,
    output_index: int,
    text: str,
) -> SSEEvent:
    data = {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": build_synthetic_message_item(item_id, text),
    }
    return SSEEvent(event="response.output_item.done", data=data)


def build_synthetic_message_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def build_synthetic_message_stream_events(
    item_id: str,
    output_index: int,
    text: str,
) -> list[SSEEvent]:
    return [
        build_synthetic_output_item_added(item_id, output_index),
        build_synthetic_content_part_added(item_id, output_index, 0),
        build_synthetic_output_text_delta(item_id, output_index, 0, text),
        build_synthetic_output_text_done(item_id, output_index, 0, text),
        build_synthetic_content_part_done(item_id, output_index, 0, text),
        build_synthetic_output_item_done(item_id, output_index, text),
    ]


def transform_completed_response(
    completed_event_data: dict[str, Any],
    reasoning_mode: str,
    synthetic_reasoning_message: dict[str, Any] | None,
    strip_reasoning: bool,
) -> dict[str, Any]:
    if reasoning_mode == "pass_through":
        return completed_event_data

    transformed = deepcopy(completed_event_data)
    response = transformed.get("response")
    if not isinstance(response, dict):
        return transformed

    output = response.get("output")
    if not isinstance(output, list):
        return transformed

    if strip_reasoning:
        output = [item for item in output if item.get("type") != "reasoning"]

    if reasoning_mode in {"think_block", "plain"} and synthetic_reasoning_message:
        output.insert(0, synthetic_reasoning_message)

    response["output"] = output
    return transformed


def transform_non_streaming_response(
    response: dict[str, Any],
    reasoning_mode: str,
    diagnostics_config: dict[str, Any],
    routing_metadata: dict[str, Any],
    strip_reasoning: bool,
) -> dict[str, Any]:
    if reasoning_mode == "pass_through":
        transformed = deepcopy(response)
    else:
        transformed = deepcopy(response)
        output = transformed.get("output")
        if isinstance(output, list):
            reasoning_text = _extract_reasoning_text_from_output(output)
            filtered_output = (
                [item for item in output if item.get("type") != "reasoning"]
                if strip_reasoning
                else output
            )
            if reasoning_text and reasoning_mode in {"think_block", "plain"}:
                text = (
                    f"<think>\n{reasoning_text}\n</think>\n\n"
                    if reasoning_mode == "think_block"
                    else f"{reasoning_text}\n\n"
                )
                filtered_output.insert(
                    0,
                    build_synthetic_message_item(SYNTHETIC_REASONING_ITEM_ID, text),
                )
            transformed["output"] = filtered_output

    if diagnostics_config.get("enabled"):
        transformed = _add_non_streaming_diagnostics(
            transformed,
            reasoning_mode,
            diagnostics_config,
            routing_metadata,
        )
    return transformed


def _add_non_streaming_diagnostics(
    response: dict[str, Any],
    reasoning_mode: str,
    diagnostics_config: dict[str, Any],
    routing_metadata: dict[str, Any],
) -> dict[str, Any]:
    output = response.setdefault("output", [])
    if not isinstance(output, list):
        return response

    text = build_diagnostics_text(
        source_api="responses",
        reasoning_mode=reasoning_mode,
        stream_insertion_strategy="transform_reasoning_events",
        selected_skill=routing_metadata.get("selected_skill"),
        confidence=routing_metadata.get("confidence"),
        manual_skill=routing_metadata.get("manual_skill"),
        format=diagnostics_config.get("format", "visible_block"),
        reasoning_present=_output_has_reasoning(output),
        reasoning_tokens=_extract_reasoning_tokens({"response": response}),
    )
    placement = diagnostics_config.get("placement", "start")
    if placement in {"start", "both"}:
        output.insert(0, build_synthetic_message_item(SYNTHETIC_DIAGNOSTICS_ITEM_ID, text))
    if placement in {"end", "both"}:
        output.append(
            build_synthetic_message_item(f"{SYNTHETIC_DIAGNOSTICS_ITEM_ID}_end", text)
        )
    return response


def build_diagnostics_text(
    source_api: str,
    reasoning_mode: str,
    stream_insertion_strategy: str,
    selected_skill: str | None,
    confidence: int | None,
    manual_skill: str | None,
    format: str,
    reasoning_present: bool | None = None,
    reasoning_tokens: int | None = None,
) -> str:
    parts = [
        f"source_api={source_api}",
        f"selected_skill={selected_skill}",
        f"confidence={confidence}",
        f"manual_skill={manual_skill}",
        f"reasoning_mode={reasoning_mode}",
        f"stream_strategy={stream_insertion_strategy}",
    ]
    if reasoning_present is not None:
        parts.append(f"reasoning_present={reasoning_present}")
    if reasoning_tokens is not None:
        parts.append(f"reasoning_tokens={reasoning_tokens}")

    body = "; ".join(parts)
    if format == "html_comment":
        return f"<!-- diagnostics: {body} -->"
    return f"> [diagnostics] {body}"


def _nested_type(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if not isinstance(value, dict):
        return None
    item_type = value.get("type")
    return item_type if isinstance(item_type, str) else None


def _extract_reasoning_tokens(completed_data: dict[str, Any]) -> int | None:
    response = completed_data.get("response", completed_data)
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        return None
    tokens = details.get("reasoning_tokens")
    return tokens if isinstance(tokens, int) else None


def _extract_reasoning_text_from_output(output: list[dict[str, Any]]) -> str:
    parts = []
    for item in output:
        if item.get("type") != "reasoning":
            continue
        parts.extend(_extract_text_from_item(item))
    return "".join(parts)


def _extract_text_from_item(item: dict[str, Any]) -> list[str]:
    texts = []
    content = item.get("content", [])
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)

    summary = item.get("summary", [])
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)

    text = item.get("text")
    if isinstance(text, str):
        texts.append(text)
    return texts


def _output_has_reasoning(output: list[dict[str, Any]]) -> bool:
    return any(item.get("type") == "reasoning" for item in output)
