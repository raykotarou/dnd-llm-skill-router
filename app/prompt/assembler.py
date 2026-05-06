from pathlib import Path

from app.skills.models import Skill


CONSISTENCY_LENS_SKILLS = {"story", "lore", "template", "analysis"}

BASE_SYSTEM_PROMPT = """Ты работаешь как локальная модель, подключенная к Obsidian Copilot через Skill Router.

Следуй выбранному skill и общим правилам ответа.

Не запускай несколько skill-режимов одновременно.

История чата используется как контекст, но текущая задача определяется последним пользовательским запросом."""


def assemble_enriched_messages(
    raw_messages: list[dict],
    base_system_prompt: str,
    answer_rules: str,
    skill_prompt: str,
    consistency_lens: str | None,
) -> list[dict]:
    base_prompt = base_system_prompt.strip() or BASE_SYSTEM_PROMPT
    enriched_messages = [
        {"role": "system", "content": base_prompt},
        {"role": "system", "content": answer_rules},
        {"role": "system", "content": skill_prompt},
    ]

    if consistency_lens:
        enriched_messages.append({"role": "system", "content": consistency_lens})

    enriched_messages.extend(raw_messages)
    return enriched_messages


def assemble_enriched_prompt(
    skill: Skill,
    shared_answer_rules_path: str | Path,
    consistency_lens_path: str | Path | None = None,
    include_consistency_lens: bool | None = None,
) -> str:
    prompt_parts = [
        Path(shared_answer_rules_path).read_text(encoding="utf-8").strip(),
    ]

    should_include_lens = (
        skill.id in CONSISTENCY_LENS_SKILLS
        if include_consistency_lens is None
        else include_consistency_lens
    )

    if should_include_lens and consistency_lens_path is not None:
        consistency_lens = Path(consistency_lens_path).read_text(encoding="utf-8").strip()
        if consistency_lens:
            prompt_parts.append(consistency_lens)

    prompt_parts.append(skill.content.strip())
    return "\n\n---\n\n".join(prompt_parts)
