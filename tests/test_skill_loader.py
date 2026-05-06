import pytest

from app.skills.loader import load_skill, load_skills


def test_loads_story_skill() -> None:
    skill = load_skill("story")

    assert skill.id == "story"
    assert skill.name == "Story Skill"
    assert "# Skill: Story" in skill.content


def test_loads_all_mvp_skills() -> None:
    skills = load_skills()

    assert sorted(skills) == ["analysis", "lore", "rules", "story", "template"]


def test_reads_skill_frontmatter() -> None:
    skill = load_skill("analysis")

    assert skill.id == "analysis"
    assert skill.name == "Analysis Skill"
    assert skill.description == "Поиск противоречий, логических дыр и несостыковок."


def test_shared_directory_is_not_loaded_as_skill() -> None:
    skills = load_skills()

    assert "_shared" not in skills


def test_missing_skill_raises_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_skill("missing")
