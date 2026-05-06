from app.graph.nodes import rank_skills, select_skill_or_clarify
from app.routing.schemas import SkillRank


class FakeLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def call_router_model(self, messages: list[dict]) -> dict:
        self.calls += 1
        return {}


def _state(**overrides):
    state = {
        "raw_request": {},
        "messages": [],
        "latest_user_message": "",
        "manual_skill": None,
        "ranked_skills": [],
        "selected_skill": None,
        "confidence": None,
        "needs_clarification": False,
        "clarification_message": None,
        "skill_prompt": None,
        "answer_rules": None,
        "consistency_lens": None,
        "enriched_messages": [],
        "lm_response": None,
    }
    state.update(overrides)
    return state


def test_manual_skill_has_absolute_priority() -> None:
    result = select_skill_or_clarify(
        _state(
            manual_skill="analysis",
            ranked_skills=[SkillRank(skill="story", confidence=99)],
        ),
        confidence_threshold=80,
    )

    assert result["selected_skill"] == "analysis"
    assert result["confidence"] == 100
    assert result["needs_clarification"] is False


def test_manual_skill_skips_router_model_call() -> None:
    lm_client = FakeLMClient()

    rank_skills(
        _state(manual_skill="rules"),
        lm_client=lm_client,
        max_ranked_skills=3,
    )

    assert lm_client.calls == 0


def test_selects_top_ranked_skill_above_threshold() -> None:
    result = select_skill_or_clarify(
        _state(ranked_skills=[SkillRank(skill="story", confidence=80)]),
        confidence_threshold=80,
    )

    assert result["selected_skill"] == "story"
    assert result["confidence"] == 80
    assert result["needs_clarification"] is False


def test_requests_clarification_below_threshold() -> None:
    result = select_skill_or_clarify(
        _state(
            ranked_skills=[
                SkillRank(skill="story", confidence=79),
                SkillRank(skill="analysis", confidence=68),
                SkillRank(skill="lore", confidence=61),
            ]
        ),
        confidence_threshold=80,
    )

    assert result["selected_skill"] is None
    assert result["confidence"] == 79
    assert result["needs_clarification"] is True
    assert result["clarification_message"] is not None
    assert "Я не уверен, какой режим лучше использовать." in result["clarification_message"]
    assert "`story` — 79%" in result["clarification_message"]
    assert "`!story`" in result["clarification_message"]
    assert "`!analysis`" in result["clarification_message"]
    assert "`!lore`" in result["clarification_message"]
