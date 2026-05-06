import json

from app.routing.skill_ranker import build_router_messages, parse_router_response


def _router_response(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


def test_parse_router_response_sorts_and_limits_ranked_skills() -> None:
    response = _router_response(
        json.dumps(
            {
                "ranked_skills": [
                    {"skill": "story", "confidence": 70, "reason": "scene"},
                    {"skill": "analysis", "confidence": 95, "reason": "check"},
                    {"skill": "lore", "confidence": 80, "reason": "world"},
                ]
            }
        )
    )

    result = parse_router_response(response, max_ranked_skills=2)

    assert [skill.skill for skill in result.ranked_skills] == ["analysis", "lore"]
    assert result.selected_skill == "analysis"
    assert result.confidence == 95
    assert result.needs_clarification is False
    assert result.source == "llm_router"


def test_parse_router_response_handles_invalid_json() -> None:
    result = parse_router_response(_router_response("not json"), max_ranked_skills=3)

    assert result.ranked_skills == []
    assert result.selected_skill is None
    assert result.confidence is None
    assert result.needs_clarification is True
    assert result.clarification_message is not None
    assert result.source == "clarification"


def test_parse_router_response_handles_invalid_schema() -> None:
    response = _router_response(
        json.dumps({"ranked_skills": [{"skill": "story", "confidence": 101}]})
    )

    result = parse_router_response(response, max_ranked_skills=3)

    assert result.needs_clarification is True
    assert result.selected_skill is None


def test_parse_router_response_accepts_classification_fallback() -> None:
    response = _router_response(json.dumps({"classification": "story"}))

    result = parse_router_response(response, max_ranked_skills=3)

    assert result.selected_skill == "story"
    assert result.confidence == 100
    assert result.needs_clarification is False
    assert result.ranked_skills[0].skill == "story"


def test_build_router_messages_does_not_embed_full_chat_history() -> None:
    huge_context = "A" * 20_000
    latest = "B" * 10_000

    messages = [
        {"role": "system", "content": huge_context},
        {"role": "user", "content": latest},
    ]

    router_messages = build_router_messages(messages, latest)
    user_prompt = router_messages[1]["content"]

    assert len(user_prompt) < 7000
    assert "...[truncated]..." in user_prompt
    assert huge_context not in user_prompt
    assert latest not in user_prompt
