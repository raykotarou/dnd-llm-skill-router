from loguru import logger

from app.config.settings import LoggingSettings
from app.graph.nodes import log_routing_decision
from app.routing.schemas import SkillRank


def _state(**overrides):
    state = {
        "raw_request": {
            "model": "dnd-skill-router",
            "messages": [{"role": "user", "content": "Придумай сцену"}],
        },
        "messages": [{"role": "user", "content": "Придумай сцену"}],
        "latest_user_message": "Придумай сцену",
        "manual_skill": None,
        "ranked_skills": [SkillRank(skill="story", confidence=91)],
        "selected_skill": "story",
        "confidence": 91,
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


def _settings(debug_full_payload: bool) -> LoggingSettings:
    return LoggingSettings(
        level="INFO",
        log_file="./logs/router.log",
        debug_full_payload=debug_full_payload,
    )


def test_log_routing_decision_does_not_log_full_payload_by_default() -> None:
    records = []
    handler_id = logger.add(records.append, format="{message}")
    try:
        log_routing_decision(_state(), _settings(debug_full_payload=False))
    finally:
        logger.remove(handler_id)

    message = records[0]
    assert "routing_decision" not in message
    assert "latest_user_message" in message
    assert "raw_request" not in message
    assert "messages" not in message


def test_log_routing_decision_can_include_full_payload_in_debug_mode() -> None:
    records = []
    handler_id = logger.add(records.append, format="{message}")
    try:
        log_routing_decision(_state(), _settings(debug_full_payload=True))
    finally:
        logger.remove(handler_id)

    message = records[0]
    assert "raw_request" in message
    assert "messages" in message
