SUPPORTED_MANUAL_SKILLS = {
    "!story": "story",
    "!analysis": "analysis",
    "!template": "template",
    "!lore": "lore",
    "!rules": "rules",
}


def detect_manual_skill_command(message: str) -> str | None:
    stripped_message = message.lstrip()
    command = stripped_message.split(maxsplit=1)[0] if stripped_message else ""
    return SUPPORTED_MANUAL_SKILLS.get(command.lower())


def strip_manual_skill_command(message: str) -> str:
    stripped_message = message.lstrip()
    command, separator, rest = stripped_message.partition(" ")

    if command.lower() not in SUPPORTED_MANUAL_SKILLS:
        return message

    return rest.lstrip() if separator else ""
