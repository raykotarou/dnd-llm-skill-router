from pathlib import Path
from typing import Any

import yaml

from app.skills.models import Skill


DEFAULT_SKILLS_DIR = "./skills"


def load_skill(skill_id: str, skills_dir: str | Path = DEFAULT_SKILLS_DIR) -> Skill:
    skill_path = _resolve_path(skills_dir) / f"{skill_id}.md"
    return _load_skill_file(skill_path)


def load_skills(directory: str | Path = DEFAULT_SKILLS_DIR) -> dict[str, Skill]:
    skills_dir = _resolve_path(directory)
    skills: dict[str, Skill] = {}

    for skill_file in sorted(skills_dir.glob("*.md")):
        skill = _load_skill_file(skill_file)
        skills[skill.id] = skill

    return skills


def load_shared_prompt(path: str | Path) -> str:
    return _resolve_path(path).read_text(encoding="utf-8").strip()


def _load_skill_file(path: str | Path) -> Skill:
    skill_path = Path(path)
    raw_content = skill_path.read_text(encoding="utf-8")
    metadata, content = _split_frontmatter(raw_content, skill_path)

    return Skill(
        id=str(metadata["id"]),
        name=str(metadata["name"]),
        description=str(metadata["description"]),
        content=content.strip(),
        path=str(skill_path),
    )


def _split_frontmatter(raw_content: str, path: Path) -> tuple[dict[str, Any], str]:
    if not raw_content.startswith("---\n"):
        raise ValueError(f"Skill file must start with YAML frontmatter: {path}")

    try:
        _, frontmatter, content = raw_content.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"Skill file has invalid YAML frontmatter: {path}") from exc

    metadata = yaml.safe_load(frontmatter)
    if not isinstance(metadata, dict):
        raise ValueError(f"Skill frontmatter must be a YAML mapping: {path}")

    for field in ("id", "name", "description"):
        if field not in metadata:
            raise ValueError(f"Skill frontmatter missing required field `{field}`: {path}")

    return metadata, content


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    project_root = Path(__file__).resolve().parents[2]
    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate

    return candidate
