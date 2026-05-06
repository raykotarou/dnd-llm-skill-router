from app.config.settings import LoggingSettings, SkillsSettings
from app.graph.pipeline import build_pipeline


class FakeLMClient:
    def __init__(self, router_confidence: int = 91) -> None:
        self.router_calls = 0
        self.main_calls = 0
        self.main_messages = None
        self.router_confidence = router_confidence

    def call_router_model(self, messages: list[dict]) -> dict:
        self.router_calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"ranked_skills":[{"skill":"story",'
                            f'"confidence":{self.router_confidence},'
                            '"reason":"story request"}]}'
                        )
                    }
                }
            ]
        }

    def call_main_model(self, messages: list[dict], raw_request: dict) -> dict:
        self.main_calls += 1
        self.main_messages = messages
        return {
            "id": "chatcmpl-main",
            "object": "chat.completion",
            "model": "qwen-main-35b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }


def _pipeline(lm_client: FakeLMClient):
    return build_pipeline(
        confidence_threshold=80,
        max_ranked_skills=3,
        skills_settings=SkillsSettings(
            directory="./skills",
            default_skill="story",
            shared_answer_rules="./skills/_shared/answer_rules.md",
            consistency_lens="./skills/_shared/consistency_lens.md",
        ),
        logging_settings=LoggingSettings(
            level="INFO",
            log_file="./logs/router.log",
            debug_full_payload=False,
        ),
        lm_client=lm_client,
    )


def _initial_state(raw_request: dict) -> dict:
    return {
        "raw_request": raw_request,
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


def test_guardrail_preserves_original_message_history() -> None:
    lm_client = FakeLMClient(router_confidence=91)
    raw_messages = [
        {"role": "user", "content": "Первый запрос"},
        {"role": "assistant", "content": "Ответ"},
        {"role": "user", "content": "Придумай сцену"},
    ]

    state = _pipeline(lm_client).invoke(
        _initial_state({"model": "dnd-skill-router", "messages": raw_messages})
    )

    assert state["messages"] == raw_messages
    assert state["enriched_messages"][-3:] == raw_messages


def test_guardrail_routes_by_latest_user_message() -> None:
    lm_client = FakeLMClient(router_confidence=91)
    raw_messages = [
        {"role": "user", "content": "!analysis Старый запрос"},
        {"role": "assistant", "content": "Ответ"},
        {"role": "user", "content": "Придумай сцену"},
    ]

    state = _pipeline(lm_client).invoke(
        _initial_state({"model": "dnd-skill-router", "messages": raw_messages})
    )

    assert state["latest_user_message"] == "Придумай сцену"
    assert state["manual_skill"] is None
    assert lm_client.router_calls == 1


def test_guardrail_does_not_call_main_model_on_low_confidence() -> None:
    lm_client = FakeLMClient(router_confidence=79)

    state = _pipeline(lm_client).invoke(
        _initial_state(
            {
                "model": "dnd-skill-router",
                "messages": [{"role": "user", "content": "Помоги с этим"}],
            }
        )
    )

    assert state["needs_clarification"] is True
    assert lm_client.main_calls == 0


def test_guardrail_does_not_call_router_model_for_manual_command() -> None:
    lm_client = FakeLMClient(router_confidence=91)

    state = _pipeline(lm_client).invoke(
        _initial_state(
            {
                "model": "dnd-skill-router",
                "messages": [{"role": "user", "content": "!story Придумай сцену"}],
            }
        )
    )

    assert state["manual_skill"] == "story"
    assert lm_client.router_calls == 0
