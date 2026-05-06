from app.routing.manual_commands import (
    detect_manual_skill_command,
    strip_manual_skill_command,
)


def test_detects_manual_skill_command_at_start() -> None:
    assert (
        detect_manual_skill_command("!analysis Проверь сцену на несостыковки")
        == "analysis"
    )


def test_detects_all_supported_manual_commands() -> None:
    assert detect_manual_skill_command("!story Придумай сцену") == "story"
    assert detect_manual_skill_command("!analysis Проверь сцену") == "analysis"
    assert detect_manual_skill_command("!template Сделай шаблон") == "template"
    assert detect_manual_skill_command("!lore Опиши королевство") == "lore"
    assert detect_manual_skill_command("!rules Как работает grapple") == "rules"


def test_ignores_command_not_at_start() -> None:
    assert detect_manual_skill_command("Текст !story внутри сообщения") is None


def test_returns_none_without_manual_command() -> None:
    assert detect_manual_skill_command("Придумай сцену в таверне") is None


def test_strips_manual_skill_command() -> None:
    assert strip_manual_skill_command("!analysis Проверь сцену") == "Проверь сцену"


def test_strips_manual_skill_command_with_leading_spaces() -> None:
    assert strip_manual_skill_command("  !lore   Опиши королевство") == "Опиши королевство"


def test_strip_returns_empty_string_for_command_only() -> None:
    assert strip_manual_skill_command("!rules") == ""


def test_strip_keeps_message_without_command() -> None:
    message = "Проверь сцену !analysis"
    assert strip_manual_skill_command(message) == message
