from typing import TypedDict

from app.routing.schemas import SkillRank


class RouterState(TypedDict):
    raw_request: dict
    messages: list[dict]
    latest_user_message: str
    manual_skill: str | None
    ranked_skills: list[SkillRank]
    selected_skill: str | None
    confidence: int | None
    needs_clarification: bool
    clarification_message: str | None
    skill_prompt: str | None
    answer_rules: str | None
    consistency_lens: str | None
    enriched_messages: list[dict]
    lm_response: dict | None
