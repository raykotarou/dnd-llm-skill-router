import json
from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError

from app.routing.schemas import RoutingResult, SkillRank


SUPPORTED_SKILLS = [
    {
        "skill": "story",
        "description": "Генерация сцен, сюжетов, NPC, диалогов и описаний",
    },
    {
        "skill": "analysis",
        "description": "Поиск противоречий, логических дыр и несостыковок",
    },
    {
        "skill": "template",
        "description": "Создание шаблонов, структур, таблиц и форматов",
    },
    {
        "skill": "lore",
        "description": "Развитие мира, истории, фракций, культур и географии",
    },
    {
        "skill": "rules",
        "description": "Работа с правилами DnD, механиками, проверками и статблоками",
    },
]

SUPPORTED_SKILL_IDS = {skill["skill"] for skill in SUPPORTED_SKILLS}

MAX_ROUTER_LATEST_MESSAGE_CHARS = 4000
MAX_ROUTER_CONTEXT_MESSAGES = 4
MAX_ROUTER_CONTEXT_MESSAGE_CHARS = 700


ROUTER_SYSTEM_PROMPT = f"""Ты классификатор skill'ов.
Твоя задача — выбрать наиболее подходящий skill для последнего пользовательского сообщения.
История чата может использоваться как контекст.
Skill всегда определяется заново по последнему пользовательскому сообщению.
Верни только JSON.
Не используй Markdown.
Не добавляй пояснения вне JSON.

Поддерживаемые skills:
{json.dumps(SUPPORTED_SKILLS, ensure_ascii=False, indent=2)}

Ожидаемый JSON:
{{
  "ranked_skills": [
    {{
      "skill": "story",
      "confidence": 91,
      "reason": "Запрос просит придумать сцену или сюжетный элемент"
    }}
  ]
}}

Не возвращай JSON вида {{"classification": "story"}}.
Всегда используй ключ ranked_skills."""


def build_router_messages(
    messages: list[dict],
    latest_user_message: str,
) -> list[dict[str, str]]:
    compact_history = _compact_recent_history(messages)
    latest_for_router = _middle_truncate(
        latest_user_message,
        MAX_ROUTER_LATEST_MESSAGE_CHARS,
    )

    return [
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Краткий контекст последних сообщений, если он полезен:\n"
                f"{json.dumps(compact_history, ensure_ascii=False)}\n\n"
                "Последнее пользовательское сообщение для классификации:\n"
                f"{latest_for_router}\n\n"
                "Верни только JSON без Markdown-обёртки."
            ),
        },
    ]


def _compact_recent_history(messages: list[dict]) -> list[dict[str, str]]:
    compact_messages = []
    for message in messages[-MAX_ROUTER_CONTEXT_MESSAGES:]:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if not isinstance(role, str):
            role = "unknown"
        if not isinstance(content, str):
            content = str(content)

        compact_messages.append(
            {
                "role": role,
                "content": _middle_truncate(
                    content,
                    MAX_ROUTER_CONTEXT_MESSAGE_CHARS,
                ),
            }
        )
    return compact_messages


def _middle_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    keep = max_chars - len("\n...[truncated]...\n")
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def parse_router_response(
    router_response: dict[str, Any],
    max_ranked_skills: int,
) -> RoutingResult:
    response_text = extract_router_response_text(router_response)

    try:
        raw_result = json.loads(response_text)
        ranked_skills = _extract_ranked_skills(raw_result)
    except (AttributeError, JSONDecodeError, TypeError, ValidationError):
        return _clarification_result()

    ranked_skills = sorted(
        ranked_skills,
        key=lambda skill_rank: skill_rank.confidence,
        reverse=True,
    )[:max_ranked_skills]
    selected_skill = ranked_skills[0].skill if ranked_skills else None
    confidence = ranked_skills[0].confidence if ranked_skills else None

    return RoutingResult(
        ranked_skills=ranked_skills,
        selected_skill=selected_skill,
        confidence=confidence,
        needs_clarification=selected_skill is None,
        clarification_message=(
            "Уточните, какой тип помощи нужен: сцена, анализ, шаблон, лор или правила?"
            if selected_skill is None
            else None
        ),
        source="llm_router",
    )


def extract_router_response_text(router_response: dict[str, Any]) -> str:
    choices = router_response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def _extract_ranked_skills(raw_result: dict[str, Any]) -> list[SkillRank]:
    if "ranked_skills" in raw_result:
        return [
            SkillRank.model_validate(skill_rank)
            for skill_rank in raw_result.get("ranked_skills", [])
        ]

    classification = raw_result.get("classification")
    if isinstance(classification, str) and classification in SUPPORTED_SKILL_IDS:
        return [
            SkillRank(
                skill=classification,
                confidence=100,
                reason="Router returned a single skill classification.",
            )
        ]

    return []


def _clarification_result() -> RoutingResult:
    return RoutingResult(
        ranked_skills=[],
        selected_skill=None,
        confidence=None,
        needs_clarification=True,
        clarification_message=(
            "Не удалось уверенно определить подходящий skill. "
            "Уточните, что нужно сделать: создать сцену, проверить несостыковки, "
            "подготовить шаблон, развить лор или разобрать правила?"
        ),
        source="clarification",
    )
