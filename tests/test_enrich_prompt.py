import pytest

from app.graph.nodes import enrich_prompt
from app.prompt.assembler import BASE_SYSTEM_PROMPT


def _state(**overrides):
    state = {
        "raw_request": {},
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
        "enriched_messages": [],
        "lm_response": None,
    }
    state.update(overrides)
    return state


def test_enrich_prompt_adds_all_instruction_layers_before_original_messages() -> None:
    result = enrich_prompt(_state())

    assert result["enriched_messages"] == [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "system", "content": "# Общие правила ответа"},
        {"role": "system", "content": "# Skill: Story"},
        {"role": "system", "content": "# Линза согласованности"},
        {"role": "user", "content": "Придумай сцену"},
    ]


def test_enrich_prompt_skips_enrichment_for_clarification() -> None:
    messages = [{"role": "user", "content": "Непонятный запрос"}]

    result = enrich_prompt(
        _state(
            messages=messages,
            needs_clarification=True,
            skill_prompt=None,
            answer_rules=None,
            consistency_lens=None,
        )
    )

    assert result["enriched_messages"] == messages


def test_enrich_prompt_requires_loaded_answer_rules() -> None:
    with pytest.raises(ValueError, match="`answer_rules` is not loaded"):
        enrich_prompt(_state(answer_rules=None))


def test_enrich_prompt_requires_loaded_skill_prompt() -> None:
    with pytest.raises(ValueError, match="`skill_prompt` is not loaded"):
        enrich_prompt(_state(skill_prompt=None))
