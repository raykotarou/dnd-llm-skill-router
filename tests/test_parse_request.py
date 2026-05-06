import pytest

from app.graph.nodes import (
    detect_manual_skill_command,
    extract_latest_user_message,
    parse_request,
)


def _state(raw_request: dict) -> dict:
    return {
        "raw_request": raw_request,
        "messages": [],
        "latest_user_message": "",
        "manual_skill": None,
        "ranked_skills": [],
        "selected_skill": None,
        "confidence": None,
        "needs_clarification": False,
        "clarification_message": None,
        "enriched_messages": [],
        "lm_response": None,
    }


def test_parse_request_extracts_messages() -> None:
    result = parse_request(
        _state(
            {
                "messages": [
                    {"role": "user", "content": "Первый запрос"},
                    {"role": "assistant", "content": "Ответ"},
                    {"role": "user", "content": "Последний запрос"},
                ]
            }
        )
    )

    assert result["messages"] == [
        {"role": "user", "content": "Первый запрос"},
        {"role": "assistant", "content": "Ответ"},
        {"role": "user", "content": "Последний запрос"},
    ]
    assert result["enriched_messages"] == result["messages"]


def test_extract_latest_user_message_uses_last_user_message_only() -> None:
    result = extract_latest_user_message(
        _state({})
        | {
            "messages": [
                {"role": "user", "content": "Первый запрос"},
                {"role": "assistant", "content": "Ответ"},
                {"role": "user", "content": "Последний запрос"},
            ]
        }
    )

    assert result["latest_user_message"] == "Последний запрос"


def test_detect_manual_skill_command_from_latest_user_message() -> None:
    result = detect_manual_skill_command(
        _state({}) | {"latest_user_message": "!analysis Проверь сцену"}
    )

    assert result["manual_skill"] == "analysis"


def test_detect_manual_skill_command_ignores_command_inside_message() -> None:
    result = detect_manual_skill_command(
        _state({}) | {"latest_user_message": "Проверь сцену !analysis"}
    )

    assert result["manual_skill"] is None


def test_parse_request_requires_messages() -> None:
    with pytest.raises(ValueError, match="`messages` is required"):
        parse_request(_state({}))


def test_parse_request_requires_messages_list() -> None:
    with pytest.raises(ValueError, match="`messages` must be a list"):
        parse_request(_state({"messages": "not-a-list"}))
