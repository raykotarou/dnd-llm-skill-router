from datetime import datetime
import time
import uuid
from typing import Any

from loguru import logger

from app.config.settings import LoggingSettings, SkillsSettings
from app.graph.state import RouterState
from app.prompt.assembler import (
    BASE_SYSTEM_PROMPT,
    CONSISTENCY_LENS_SKILLS,
    assemble_enriched_messages,
)
from app.routing.manual_commands import detect_manual_skill_command as detect_command
from app.routing.schemas import SkillRank
from app.routing.skill_ranker import build_router_messages, parse_router_response
from app.skills.loader import load_shared_prompt, load_skill


SKILL_CLARIFICATION_HINTS = {
    "story": "Подходит, если нужно развить сцену, сюжет или описание.",
    "analysis": "Подходит, если нужно найти противоречия или логические дыры.",
    "template": "Подходит, если нужно создать структуру, шаблон, таблицу или формат.",
    "lore": "Подходит, если нужно расширить или уточнить лор мира.",
    "rules": "Подходит, если нужно разобрать правила, механику, проверки или статблоки.",
}


def parse_request(state: RouterState) -> RouterState:
    raw_request = state["raw_request"]
    messages = raw_request.get("messages")

    if messages is None:
        raise ValueError("Invalid OpenAI chat completion payload: `messages` is required.")

    if not isinstance(messages, list):
        raise ValueError(
            "Invalid OpenAI chat completion payload: `messages` must be a list."
        )

    return {
        "messages": messages,
        "enriched_messages": messages,
    }


def extract_latest_user_message(state: RouterState) -> RouterState:
    return {
        "latest_user_message": _latest_user_message(state["messages"]),
    }


def detect_manual_skill_command(state: RouterState) -> RouterState:
    return {
        "manual_skill": detect_command(state["latest_user_message"]),
    }


def rank_skills(
    state: RouterState,
    lm_client,
    max_ranked_skills: int,
) -> RouterState:
    if state["manual_skill"] is not None:
        return {"ranked_skills": state["ranked_skills"]}

    router_messages = build_router_messages(
        messages=state["messages"],
        latest_user_message=state["latest_user_message"],
    )
    logger.info("Calling LM Studio router model")
    router_response = lm_client.call_router_model(router_messages)
    logger.info("LM Studio router model returned response")
    routing_result = parse_router_response(router_response, max_ranked_skills)
    logger.info(
        "Router parsed result: selected_skill={}, confidence={}, needs_clarification={}",
        routing_result.selected_skill,
        routing_result.confidence,
        routing_result.needs_clarification,
    )

    return {
        "ranked_skills": routing_result.ranked_skills,
        "selected_skill": routing_result.selected_skill,
        "confidence": routing_result.confidence,
        "needs_clarification": routing_result.needs_clarification,
        "clarification_message": routing_result.clarification_message,
    }


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)

    return ""


def select_skill_or_clarify(state: RouterState, confidence_threshold: int) -> RouterState:
    manual_skill = state["manual_skill"]
    if manual_skill is not None:
        return {
            "selected_skill": manual_skill,
            "confidence": 100,
            "needs_clarification": False,
            "clarification_message": None,
        }

    ranked_skills = state["ranked_skills"]
    top_skill = ranked_skills[0] if ranked_skills else None

    if top_skill is not None and top_skill.confidence >= confidence_threshold:
        return {
            "selected_skill": top_skill.skill,
            "confidence": top_skill.confidence,
            "needs_clarification": False,
            "clarification_message": None,
        }

    return {
        "selected_skill": None,
        "confidence": top_skill.confidence if top_skill is not None else None,
        "needs_clarification": True,
        "clarification_message": build_clarification_message(ranked_skills),
    }


select_skill = select_skill_or_clarify


def log_routing_decision(
    state: RouterState,
    logging_settings: LoggingSettings,
) -> RouterState:
    decision: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "latest_user_message": state["latest_user_message"],
        "manual_skill": state["manual_skill"],
        "ranked_skills": [
            {
                "skill": skill_rank.skill,
                "confidence": skill_rank.confidence,
            }
            for skill_rank in state["ranked_skills"]
        ],
        "selected_skill": state["selected_skill"],
        "needs_clarification": state["needs_clarification"],
    }

    if logging_settings.debug_full_payload:
        decision["raw_request"] = state["raw_request"]

    logger.bind(event="routing_decision").info("{}", decision)
    return {}


def load_skill_prompt(state: RouterState, skills_settings: SkillsSettings) -> RouterState:
    if state["needs_clarification"]:
        return {
            "skill_prompt": None,
            "answer_rules": None,
            "consistency_lens": None,
        }

    selected_skill = state["selected_skill"]
    if selected_skill is None:
        raise ValueError("Cannot load skill prompt: `selected_skill` is not set.")

    try:
        skill = load_skill(selected_skill, skills_settings.directory)
    except FileNotFoundError as exc:
        raise ValueError(f"Selected skill not found: {selected_skill}") from exc

    answer_rules = load_shared_prompt(skills_settings.shared_answer_rules)
    consistency_lens = (
        load_shared_prompt(skills_settings.consistency_lens)
        if selected_skill in CONSISTENCY_LENS_SKILLS
        else None
    )

    return {
        "skill_prompt": skill.content,
        "answer_rules": answer_rules,
        "consistency_lens": consistency_lens,
    }


def enrich_prompt(state: RouterState) -> RouterState:
    if state["needs_clarification"]:
        return {"enriched_messages": state["messages"]}

    answer_rules = state["answer_rules"]
    skill_prompt = state["skill_prompt"]

    if answer_rules is None:
        raise ValueError("Cannot enrich prompt: `answer_rules` is not loaded.")

    if skill_prompt is None:
        raise ValueError("Cannot enrich prompt: `skill_prompt` is not loaded.")

    return {
        "enriched_messages": assemble_enriched_messages(
            raw_messages=state["messages"],
            base_system_prompt=BASE_SYSTEM_PROMPT,
            answer_rules=answer_rules,
            skill_prompt=skill_prompt,
            consistency_lens=state["consistency_lens"],
        )
    }


def call_lm_studio(state: RouterState, lm_client) -> RouterState:
    if state["needs_clarification"]:
        logger.info("Skipping LM Studio main model because clarification is needed")
        return {}

    if state["selected_skill"] is None:
        raise ValueError("Cannot call main model: `selected_skill` is not set.")

    logger.info(
        "Calling LM Studio main model for selected_skill={}",
        state["selected_skill"],
    )
    response = lm_client.call_main_model(
        messages=state["enriched_messages"],
        raw_request=state["raw_request"],
    )
    logger.info("LM Studio main model returned response")
    return {"lm_response": response}


def build_clarification_message(ranked_skills: list[SkillRank]) -> str:
    lines = [
        "Я не уверен, какой режим лучше использовать.",
        "",
        "Возможные варианты:",
        "",
    ]

    if ranked_skills:
        for index, skill_rank in enumerate(ranked_skills, start=1):
            hint = SKILL_CLARIFICATION_HINTS.get(
                skill_rank.skill,
                "Подходит, если это ближе всего к вашему запросу.",
            )
            lines.extend(
                [
                    f"{index}. `{skill_rank.skill}` — {skill_rank.confidence}%  ",
                    f"   {hint}",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "1. `story` — для сцен, сюжетов, NPC, диалогов и описаний.",
                "2. `analysis` — для противоречий, логических дыр и несостыковок.",
                "3. `template` — для структур, шаблонов, таблиц и форматов.",
                "4. `lore` — для мира, истории, фракций, культур и географии.",
                "5. `rules` — для правил DnD, механик, проверок и статблоков.",
                "",
            ]
        )

    lines.extend(
        [
            "Выбери skill командой, например:",
            "",
        ]
    )

    command_skills = [skill_rank.skill for skill_rank in ranked_skills[:3]]
    if not command_skills:
        command_skills = ["story", "analysis", "lore"]

    lines.extend(f"`!{skill}`" for skill in command_skills)
    return "\n".join(lines)


def format_response(state: RouterState) -> RouterState:
    if state["lm_response"] is not None:
        return {}

    payload = state["raw_request"]
    model = payload.get("model", "dnd-skill-router")
    last_user_message = state["latest_user_message"]

    if state["needs_clarification"] and state["clarification_message"] is not None:
        content = state["clarification_message"]
    else:
        content = (
            "Запрос принят dnd-skill-router. "
            "История сообщений сохранена и прошла через LangGraph pipeline."
        )
        if state["selected_skill"] is not None:
            content = f"{content}\n\nВыбранный skill: {state['selected_skill']}"
        if last_user_message:
            content = f"{content}\n\nПоследний запрос: {last_user_message}"

    return {
        "lm_response": {
            "id": (
                "chatcmpl-local-clarification"
                if state["needs_clarification"]
                else f"chatcmpl-{uuid.uuid4().hex}"
            ),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    }


create_chat_completion_response = format_response
