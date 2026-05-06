from typing import Literal

from pydantic import BaseModel, Field


class SkillRank(BaseModel):
    skill: str
    confidence: int = Field(ge=0, le=100)
    reason: str | None = None


class RoutingResult(BaseModel):
    ranked_skills: list[SkillRank]
    selected_skill: str | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    needs_clarification: bool = False
    clarification_message: str | None = None
    source: Literal["manual_command", "llm_router", "clarification"]
