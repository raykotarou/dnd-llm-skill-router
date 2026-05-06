from app.adapters.sse import format_sse_event, parse_sse_event


def test_parse_event_with_event_and_data() -> None:
    event = parse_sse_event(
        'event: response.output_text.delta\ndata: {"delta":"ok"}\n\n'
    )

    assert event.event == "response.output_text.delta"
    assert event.data == {"delta": "ok"}


def test_parse_event_with_only_data_uses_type() -> None:
    event = parse_sse_event('data: {"type":"response.completed"}\n\n')

    assert event.event == "response.completed"
    assert event.data == {"type": "response.completed"}


def test_parse_event_keeps_non_json_data_as_string() -> None:
    event = parse_sse_event("data: [DONE]\n\n")

    assert event.event is None
    assert event.data == "[DONE]"


def test_format_sse_event_uses_double_newline_and_event_type() -> None:
    result = format_sse_event("response.output_text.delta", {"delta": "ok"})

    assert result == b'event: response.output_text.delta\ndata: {"delta": "ok"}\n\n'


def test_format_sse_event_handles_string_data() -> None:
    result = format_sse_event(None, "[DONE]")

    assert result == b"data: [DONE]\n\n"
