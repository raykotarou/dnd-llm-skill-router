from app.adapters.reasoning_stream_transformer import (
    ResponsesReasoningStreamTransformer,
    build_diagnostics_text,
    transform_completed_response,
    transform_non_streaming_response,
)
from app.adapters.sse import SSEEvent


def _transformer(mode: str, diagnostics: dict | None = None):
    return ResponsesReasoningStreamTransformer(
        reasoning_mode=mode,
        stream_insertion_strategy="transform_reasoning_events",
        diagnostics_config=diagnostics
        or {
            "enabled": False,
            "placement": "start",
            "format": "visible_block",
        },
        routing_metadata={
            "selected_skill": "story",
            "confidence": 100,
            "manual_skill": "story",
        },
    )


def _reasoning_delta(text: str = "мысль") -> SSEEvent:
    return SSEEvent(
        event="response.reasoning_text.delta",
        data={"type": "response.reasoning_text.delta", "delta": text},
    )


def _reasoning_done() -> SSEEvent:
    return SSEEvent(
        event="response.reasoning_text.done",
        data={"type": "response.reasoning_text.done", "text": "мысль"},
    )


def _output_delta() -> SSEEvent:
    return SSEEvent(
        event="response.output_text.delta",
        data={"type": "response.output_text.delta", "delta": "ответ"},
    )


def _completed() -> SSEEvent:
    return SSEEvent(
        event="response.completed",
        data={
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "output": [
                    {"id": "rsn", "type": "reasoning", "content": []},
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ответ"}],
                    },
                ],
                "usage": {
                    "output_tokens_details": {
                        "reasoning_tokens": 112,
                    }
                },
            },
        },
    )


def test_drop_suppresses_reasoning_delta_and_preserves_output_delta() -> None:
    transformer = _transformer("drop")

    assert transformer.transform(_reasoning_delta()) == []
    assert transformer.transform(_output_delta())[0].event == "response.output_text.delta"


def test_drop_suppresses_reasoning_items_and_strips_completed_reasoning() -> None:
    transformer = _transformer("drop")

    assert transformer.transform(
        SSEEvent(
            event="response.output_item.added",
            data={"item": {"type": "reasoning"}},
        )
    ) == []
    assert transformer.transform(
        SSEEvent(
            event="response.content_part.added",
            data={"part": {"type": "reasoning_text"}},
        )
    ) == []

    result = transformer.transform(_completed())[0].data

    output = result["response"]["output"]
    assert [item["type"] for item in output] == ["message"]
    assert result["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 112


def test_think_block_transforms_reasoning_to_output_text() -> None:
    transformer = _transformer("think_block")

    events = transformer.transform(_reasoning_delta("abc"))
    done_events = transformer.transform(_reasoning_done())

    deltas = [
        event.data["delta"]
        for event in [*events, *done_events]
        if event.event == "response.output_text.delta"
    ]
    assert deltas == ["<think>\nabc", "\n</think>\n\n"]
    assert events[0].event == "response.output_item.added"
    assert events[0].data["item"]["role"] == "assistant"


def test_think_block_completed_contains_synthetic_message() -> None:
    transformer = _transformer("think_block")
    transformer.transform(_reasoning_delta("abc"))
    transformer.transform(_reasoning_done())

    completed = transformer.transform(_completed())[-1].data
    output = completed["response"]["output"]

    assert [item["type"] for item in output] == ["message", "message"]
    assert output[0]["role"] == "assistant"
    assert output[0]["content"][0]["type"] == "output_text"
    assert "<think>\nabc\n</think>" in output[0]["content"][0]["text"]


def test_think_block_does_not_create_empty_message_without_reasoning() -> None:
    result = transform_completed_response(
        completed_event_data=_completed().data,
        reasoning_mode="think_block",
        synthetic_reasoning_message=None,
        strip_reasoning=True,
    )

    assert [item["type"] for item in result["response"]["output"]] == ["message"]


def test_plain_transforms_reasoning_without_think_tags() -> None:
    transformer = _transformer("plain")

    events = transformer.transform(_reasoning_delta("abc"))
    done_events = transformer.transform(_reasoning_done())

    deltas = [
        event.data["delta"]
        for event in [*events, *done_events]
        if event.event == "response.output_text.delta"
    ]
    assert deltas == ["abc", "\n\n"]
    assert "<think>" not in "".join(deltas)


def test_plain_completed_contains_synthetic_message_without_tags() -> None:
    transformer = _transformer("plain")
    transformer.transform(_reasoning_delta("abc"))
    transformer.transform(_reasoning_done())

    completed = transformer.transform(_completed())[-1].data
    text = completed["response"]["output"][0]["content"][0]["text"]

    assert text == "abc\n\n"


def test_pass_through_keeps_reasoning_and_completed_unchanged() -> None:
    transformer = _transformer("pass_through")
    reasoning = _reasoning_delta("abc")
    completed = _completed()

    assert transformer.transform(reasoning)[0] == reasoning
    assert transformer.transform(completed)[0].data == completed.data


def test_diagnostics_text_formats() -> None:
    visible = build_diagnostics_text(
        source_api="responses",
        reasoning_mode="drop",
        stream_insertion_strategy="transform_reasoning_events",
        selected_skill="story",
        confidence=100,
        manual_skill="story",
        format="visible_block",
    )
    comment = build_diagnostics_text(
        source_api="responses",
        reasoning_mode="drop",
        stream_insertion_strategy="transform_reasoning_events",
        selected_skill="story",
        confidence=100,
        manual_skill="story",
        format="html_comment",
    )

    assert visible.startswith("> [diagnostics]")
    assert "reasoning_mode=drop" in visible
    assert comment.startswith("<!-- diagnostics:")
    assert comment.endswith(" -->")


def test_diagnostics_start_and_end_stream_events() -> None:
    transformer = _transformer(
        "drop",
        diagnostics={"enabled": True, "placement": "both", "format": "visible_block"},
    )

    created_events = transformer.transform(
        SSEEvent(event="response.created", data={"type": "response.created"})
    )
    completed_events = transformer.transform(_completed())

    diagnostic_deltas = [
        event
        for event in [*created_events, *completed_events]
        if event.event == "response.output_text.delta"
        and "[diagnostics]" in event.data["delta"]
    ]
    assert len(diagnostic_deltas) == 2


def test_non_streaming_drop_removes_reasoning_and_preserves_usage() -> None:
    response = _completed().data["response"]

    transformed = transform_non_streaming_response(
        response=response,
        reasoning_mode="drop",
        diagnostics_config={"enabled": False},
        routing_metadata={},
        strip_reasoning=True,
    )

    assert [item["type"] for item in transformed["output"]] == ["message"]
    assert transformed["usage"]["output_tokens_details"]["reasoning_tokens"] == 112


def test_non_streaming_think_block_inserts_reasoning_message() -> None:
    response = {
        **_completed().data["response"],
        "output": [
            {
                "id": "rsn",
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "abc"}],
            },
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ответ"}],
            },
        ],
    }

    transformed = transform_non_streaming_response(
        response=response,
        reasoning_mode="think_block",
        diagnostics_config={"enabled": False},
        routing_metadata={},
        strip_reasoning=True,
    )

    assert "<think>\nabc\n</think>" in transformed["output"][0]["content"][0]["text"]
    assert [item["type"] for item in transformed["output"]] == ["message", "message"]
