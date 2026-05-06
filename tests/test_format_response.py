from app.graph.nodes import format_response


def _state(**overrides):
    state = {
        "raw_request": {"model": "dnd-skill-router"},
        "messages": [],
        "latest_user_message": "",
        "manual_skill": None,
        "ranked_skills": [],
        "selected_skill": None,
        "confidence": None,
        "needs_clarification": True,
        "clarification_message": "Уточните skill",
        "skill_prompt": None,
        "answer_rules": None,
        "consistency_lens": None,
        "enriched_messages": [],
        "lm_response": None,
    }
    state.update(overrides)
    return state


def test_format_response_returns_openai_compatible_clarification() -> None:
    result = format_response(_state())
    response = result["lm_response"]

    assert response["id"] == "chatcmpl-local-clarification"
    assert response["object"] == "chat.completion"
    assert response["model"] == "dnd-skill-router"
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Уточните skill",
    }
    assert response["choices"][0]["finish_reason"] == "stop"
    assert "usage" in response


def test_format_response_keeps_existing_lm_response() -> None:
    lm_response = {
        "id": "chatcmpl-lm-studio",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "qwen-main-35b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Ответ LM Studio"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    result = format_response(_state(lm_response=lm_response))

    assert result == {}


def test_existing_lm_response_remains_available_without_user_metadata() -> None:
    lm_response = {
        "id": "chatcmpl-lm-studio",
        "object": "chat.completion",
        "model": "qwen-main-35b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Ответ LM Studio"},
                "finish_reason": "stop",
            }
        ],
    }
    state = _state(lm_response=lm_response)

    format_response(state)

    assert state["lm_response"] == lm_response
    assert "routing_metadata" not in state["lm_response"]
