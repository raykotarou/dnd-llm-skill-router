import asyncio

from app.config.settings import LoggingSettings, SkillsSettings
from app.graph.pipeline import prepare_generation, complete_generation, build_prepare_pipeline


class FakeLMClient:
    def __init__(self, router_response=None):
        self.router_calls = 0
        self.main_calls = 0
        self.router_response = router_response or {}

    def call_router_model(self, messages: list[dict]) -> dict:
        self.router_calls += 1
        return self.router_response

    def call_main_model(self, messages: list[dict], raw_request: dict) -> dict:
        self.main_calls += 1
        return {
            "id": "chatcmpl-main",
            "object": "chat.completion",
            "model": "qwen-main-35b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Analysis response",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


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


def test_manual_skill_flow_skips_router_and_calls_main_model(monkeypatch) -> None:
    async def run_test():
        lm_client = FakeLMClient()
        monkeypatch.setattr(
            "app.graph.pipeline.LMStudioClient",
            lambda settings: lm_client,
        )
        pipeline = build_prepare_pipeline(
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

        state = pipeline.invoke(
            _initial_state(
                {
                    "model": "dnd-skill-router",
                    "messages": [
                        {
                            "role": "user",
                            "content": "!analysis Проверь эту сцену на несостыковки",
                        }
                    ],
                }
            )
        )

        assert state["manual_skill"] == "analysis"
        assert state["selected_skill"] == "analysis"
        assert state["confidence"] == 100
        assert state["needs_clarification"] is False
        assert lm_client.router_calls == 0
        # Main model not called in prepare_generation
        assert lm_client.main_calls == 0

        # Now complete the generation
        prepared = await prepare_generation({
            "model": "dnd-skill-router",
            "messages": [
                {
                    "role": "user",
                    "content": "!analysis Проверь эту сцену на несостыковки",
                }
            ],
        })
        response = await complete_generation(prepared)
        assert lm_client.main_calls == 1
        assert response["id"] == "chatcmpl-main"

    asyncio.run(run_test())


def test_automatic_routing_flow_calls_router_selects_story_and_calls_main_model(
    monkeypatch,
) -> None:
    lm_client = FakeLMClient(
        router_response={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"ranked_skills":[{"skill":"story",'
                            '"confidence":91,'
                            '"reason":"Запрос просит придумать сцену"}]}'
                        )
                    }
                }
            ]
        }
    )
    monkeypatch.setattr(
        "app.graph.pipeline.LMStudioClient",
        lambda settings: lm_client,
    )
    pipeline = build_prepare_pipeline(
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

    state = pipeline.invoke(
        _initial_state(
            {
                "model": "dnd-skill-router",
                "messages": [
                    {
                        "role": "user",
                        "content": "Придумай сцену встречи с подозрительным торговцем",
                    }
                ],
            }
        )
    )

    assert state["manual_skill"] is None
    assert lm_client.router_calls == 1
    assert state["selected_skill"] == "story"
    assert state["confidence"] == 91
    assert state["needs_clarification"] is False
    # Main model not called in prepare_generation
    assert lm_client.main_calls == 0

    # Complete generation
    prepared = asyncio.run(prepare_generation({
        "model": "dnd-skill-router",
        "messages": [
            {
                "role": "user",
                "content": "Придумай сцену встречи с подозрительным торговцем",
            }
        ],
    }))
    response = asyncio.run(complete_generation(prepared))
    assert lm_client.main_calls == 1


def test_clarification_flow_skips_main_model_and_returns_question(monkeypatch) -> None:
    lm_client = FakeLMClient(
        router_response={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"ranked_skills":[{"skill":"story",'
                            '"confidence":79,'
                            '"reason":"Недостаточно уверенности"}]}'
                        )
                    }
                }
            ]
        }
    )
    monkeypatch.setattr(
        "app.graph.pipeline.LMStudioClient",
        lambda settings: lm_client,
    )
    pipeline = build_prepare_pipeline(
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

    state = pipeline.invoke(
        _initial_state(
            {
                "model": "dnd-skill-router",
                "messages": [
                    {
                        "role": "user",
                        "content": "Помоги с этим",
                    }
                ],
            }
        )
    )

    assert state["confidence"] == 79
    assert state["selected_skill"] is None
    assert state["needs_clarification"] is True
    assert lm_client.main_calls == 0

    # Complete generation for clarification
    prepared = asyncio.run(prepare_generation({
        "model": "dnd-skill-router",
        "messages": [
            {
                "role": "user",
                "content": "Помоги с этим",
            }
        ],
    }))
    response = asyncio.run(complete_generation(prepared))
    assert response["id"] == "chatcmpl-local-clarification"
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert (
        "Я не уверен, какой режим лучше использовать."
        in response["choices"][0]["message"]["content"]
    )
