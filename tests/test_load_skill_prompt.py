from pathlib import Path

import pytest

from app.config.settings import SkillsSettings
from app.graph.nodes import load_skill_prompt


def _settings() -> SkillsSettings:
    return SkillsSettings(
        directory="./skills",
        default_skill="story",
        shared_answer_rules="./skills/_shared/answer_rules.md",
        consistency_lens="./skills/_shared/consistency_lens.md",
    )


def _state(**overrides):
    state = {
        "raw_request": {},
        "messages": [],
        "latest_user_message": "",
        "manual_skill": None,
        "ranked_skills": [],
        "selected_skill": "story",
        "confidence": 100,
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


def test_load_skill_prompt_loads_skill_answer_rules_and_lens() -> None:
    result = load_skill_prompt(_state(selected_skill="story"), _settings())

    assert result["skill_prompt"] is not None
    assert "# Skill: Story" in result["skill_prompt"]
    assert result["answer_rules"] is not None
    assert "# Общие правила ответа" in result["answer_rules"]
    assert result["consistency_lens"] is not None
    assert "# Линза согласованности" in result["consistency_lens"]


def test_load_skill_prompt_skips_loading_when_clarification_needed() -> None:
    result = load_skill_prompt(
        _state(needs_clarification=True, selected_skill=None),
        _settings(),
    )

    assert result == {
        "skill_prompt": None,
        "answer_rules": None,
        "consistency_lens": None,
    }


def test_load_skill_prompt_keeps_lens_optional_for_rules() -> None:
    result = load_skill_prompt(_state(selected_skill="rules"), _settings())

    assert result["skill_prompt"] is not None
    assert "# Skill: Rules" in result["skill_prompt"]
    assert result["answer_rules"] is not None
    assert result["consistency_lens"] is None


def test_load_skill_prompt_raises_clear_error_for_missing_skill(tmp_path: Path) -> None:
    settings = SkillsSettings(
        directory=str(tmp_path),
        default_skill="story",
        shared_answer_rules="./skills/_shared/answer_rules.md",
        consistency_lens="./skills/_shared/consistency_lens.md",
    )

    with pytest.raises(ValueError, match="Selected skill not found: missing"):
        load_skill_prompt(_state(selected_skill="missing"), settings)
