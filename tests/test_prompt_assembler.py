from app.prompt.assembler import BASE_SYSTEM_PROMPT, assemble_enriched_messages


def test_assemble_enriched_messages_preserves_original_messages() -> None:
    raw_messages = [
        {"role": "system", "content": "Obsidian system"},
        {"role": "user", "content": "Придумай сцену"},
    ]

    enriched = assemble_enriched_messages(
        raw_messages=raw_messages,
        base_system_prompt="Base system prompt",
        answer_rules="Shared answer rules",
        skill_prompt="Story skill",
        consistency_lens="Consistency lens",
    )

    assert enriched[:4] == [
        {"role": "system", "content": "Base system prompt"},
        {"role": "system", "content": "Shared answer rules"},
        {"role": "system", "content": "Story skill"},
        {"role": "system", "content": "Consistency lens"},
    ]
    assert enriched[4:] == raw_messages
    assert raw_messages == [
        {"role": "system", "content": "Obsidian system"},
        {"role": "user", "content": "Придумай сцену"},
    ]


def test_assemble_enriched_messages_skips_empty_consistency_lens() -> None:
    raw_messages = [{"role": "user", "content": "Придумай сцену"}]

    enriched = assemble_enriched_messages(
        raw_messages=raw_messages,
        base_system_prompt="Base system prompt",
        answer_rules="Shared answer rules",
        skill_prompt="Story skill",
        consistency_lens=None,
    )

    assert enriched == [
        {"role": "system", "content": "Base system prompt"},
        {"role": "system", "content": "Shared answer rules"},
        {"role": "system", "content": "Story skill"},
        {"role": "user", "content": "Придумай сцену"},
    ]


def test_proxy_system_prompts_have_priority_without_dropping_obsidian_system() -> None:
    raw_messages = [
        {"role": "system", "content": "Obsidian campaign context"},
        {"role": "user", "content": "Опиши город"},
    ]

    enriched = assemble_enriched_messages(
        raw_messages=raw_messages,
        base_system_prompt="Proxy base",
        answer_rules="Proxy answer rules",
        skill_prompt="Proxy skill",
        consistency_lens=None,
    )

    assert [message["content"] for message in enriched[:3]] == [
        "Proxy base",
        "Proxy answer rules",
        "Proxy skill",
    ]
    assert enriched[3:] == raw_messages
    assert {"role": "system", "content": "Obsidian campaign context"} in enriched


def test_base_system_prompt_is_used_when_custom_prompt_is_empty() -> None:
    enriched = assemble_enriched_messages(
        raw_messages=[{"role": "user", "content": "Опиши город"}],
        base_system_prompt="",
        answer_rules="Proxy answer rules",
        skill_prompt="Proxy skill",
        consistency_lens=None,
    )

    assert enriched[0] == {"role": "system", "content": BASE_SYSTEM_PROMPT}
    assert "Skill Router" in enriched[0]["content"]
