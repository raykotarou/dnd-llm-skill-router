import pytest

from app.adapters.responses_adapter import responses_to_canonical


def test_input_string_becomes_user_message() -> None:
    canonical = responses_to_canonical(
        {
            "model": "dnd-skill-router",
            "input": "Придумай сцену в таверне",
        }
    )

    assert canonical.source_api == "responses"
    assert canonical.messages == [
        {"role": "user", "content": "Придумай сцену в таверне"}
    ]


def test_input_array_with_string_content_becomes_messages() -> None:
    canonical = responses_to_canonical(
        {
            "model": "dnd-skill-router",
            "input": [
                {"role": "user", "content": "Придумай сцену в таверне"},
                {"role": "assistant", "content": "Готово"},
            ],
        }
    )

    assert canonical.messages == [
        {"role": "user", "content": "Придумай сцену в таверне"},
        {"role": "assistant", "content": "Готово"},
    ]


def test_content_parts_are_joined_as_text() -> None:
    canonical = responses_to_canonical(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Первая строка"},
                        {"type": "text", "text": "Вторая строка"},
                    ],
                }
            ],
        }
    )

    assert canonical.messages == [
        {"role": "user", "content": "Первая строка\nВторая строка"}
    ]


def test_developer_role_maps_to_system() -> None:
    canonical = responses_to_canonical(
        {
            "input": [
                {"role": "developer", "content": "Следуй правилам кампании."}
            ]
        }
    )

    assert canonical.messages == [
        {"role": "system", "content": "Следуй правилам кампании."}
    ]


def test_unknown_role_raises_validation_error() -> None:
    with pytest.raises(ValueError, match="unsupported role"):
        responses_to_canonical({"input": [{"role": "critic", "content": "Проверь"}]})


def test_missing_input_raises_validation_error() -> None:
    with pytest.raises(ValueError, match="`input` is required"):
        responses_to_canonical({"model": "dnd-skill-router"})


def test_stream_and_previous_response_id_are_preserved() -> None:
    canonical = responses_to_canonical(
        {
            "input": "Продолжи",
            "stream": True,
            "previous_response_id": "resp_123",
        }
    )

    assert canonical.stream is True
    assert canonical.previous_response_id == "resp_123"


def test_unsupported_fields_are_collected() -> None:
    canonical = responses_to_canonical(
        {
            "input": "Привет",
            "tools": [{"type": "mcp"}],
            "background": True,
        }
    )

    assert canonical.unsupported_fields == ["background", "tools"]
