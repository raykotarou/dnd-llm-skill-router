import json

from app.graph.nodes import rank_skills


class FakeLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def call_router_model(self, messages: list[dict]) -> dict:
        self.calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ranked_skills": [
                                    {
                                        "skill": "story",
                                        "confidence": 91,
                                        "reason": "scene request",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }


def _state(**overrides):
    state = {
        "raw_request": {},
        "messages": [{"role": "user", "content": "Придумай сцену"}],
        "latest_user_message": "Придумай сцену",
        "manual_skill": None,
        "ranked_skills": [],
        "selected_skill": None,
        "confidence": None,
        "needs_clarification": False,
        "clarification_message": None,
        "enriched_messages": [],
        "lm_response": None,
    }
    state.update(overrides)
    return state


def test_rank_skills_skips_router_model_when_manual_skill_exists() -> None:
    lm_client = FakeLMClient()

    result = rank_skills(
        _state(manual_skill="analysis"),
        lm_client=lm_client,
        max_ranked_skills=3,
    )

    assert lm_client.calls == 0
    assert result["ranked_skills"] == []


def test_rank_skills_calls_router_model_without_manual_skill() -> None:
    lm_client = FakeLMClient()

    result = rank_skills(
        _state(),
        lm_client=lm_client,
        max_ranked_skills=3,
    )

    assert lm_client.calls == 1
    assert result["ranked_skills"][0].skill == "story"
    assert result["ranked_skills"][0].confidence == 91
