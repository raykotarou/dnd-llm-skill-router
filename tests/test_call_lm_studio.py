import pytest

from app.graph.nodes import call_lm_studio, format_response


class FakeLMClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = None
        self.raw_request = None

    def call_main_model(self, messages: list[dict], raw_request: dict) -> dict:
        self.calls += 1
        self.messages = messages
        self.raw_request = raw_request
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "qwen-main-35b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Готово"},
                    "finish_reason": "stop",
                }
            ],
        }


def _state(**overrides):
    state = {
        "raw_request": {"model": "dnd-skill-router"},
        "messages": [{"role": "user", "content": "Придумай сцену"}],
        "latest_user_message": "Придумай сцену",
        "manual_skill": "story",
        "ranked_skills": [],
        "selected_skill": "story",
        "confidence": 100,
        "needs_clarification": False,
        "clarification_message": None,
        "skill_prompt": "# Skill: Story",
        "answer_rules": "# Общие правила ответа",
        "consistency_lens": "# Линза согласованности",
        "enriched_messages": [
            {"role": "system", "content": "# Skill: Story"},
            {"role": "user", "content": "Придумай сцену"},
        ],
        "lm_response": None,
    }
    state.update(overrides)
    return state


def test_call_lm_studio_skips_main_model_for_clarification() -> None:
    lm_client = FakeLMClient()

    result = call_lm_studio(
        _state(
            needs_clarification=True,
            selected_skill=None,
            clarification_message="Уточните skill",
        ),
        lm_client,
    )

    assert result == {}
    assert lm_client.calls == 0


def test_call_lm_studio_calls_main_model_when_skill_selected() -> None:
    lm_client = FakeLMClient()
    state = _state()

    result = call_lm_studio(state, lm_client)

    assert lm_client.calls == 1
    assert lm_client.messages == state["enriched_messages"]
    assert lm_client.raw_request == state["raw_request"]
    assert result["lm_response"]["id"] == "chatcmpl-test"


def test_call_lm_studio_requires_selected_skill() -> None:
    with pytest.raises(ValueError, match="`selected_skill` is not set"):
        call_lm_studio(_state(selected_skill=None), FakeLMClient())


def test_create_response_does_not_overwrite_lm_studio_response() -> None:
    state = _state(lm_response={"id": "chatcmpl-test", "object": "chat.completion"})

    assert format_response(state) == {}
