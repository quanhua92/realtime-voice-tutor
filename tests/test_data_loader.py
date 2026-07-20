"""Tests for data_loader.DataLoader."""

from __future__ import annotations

import pytest

from voice_tutor.data_loader import DataLoader


@pytest.fixture(scope="module")
def loader() -> DataLoader:
    return DataLoader("data")


def test_loads_all_six_scenarios(loader: DataLoader) -> None:
    assert len(loader.scenarios) == 6
    expected = {
        "restaurant",
        "airport",
        "workplace",
        "shopping",
        "meeting-people",
        "phone-calls",
    }
    assert set(loader.scenarios.keys()) == expected


def test_loads_all_three_vocabulary_files(loader: DataLoader) -> None:
    assert len(loader.vocabulary) == 3
    expected = {"a2-beginner", "b1-intermediate", "b2-upper"}
    assert set(loader.vocabulary.keys()) == expected


def test_each_scenario_has_required_frontmatter(loader: DataLoader) -> None:
    required_keys = {"title", "category", "tags", "difficulty"}
    for name, data in loader.scenarios.items():
        meta = data["meta"]
        missing = required_keys - meta.keys()
        assert not missing, f"{name} missing frontmatter keys: {missing}"
        assert isinstance(meta["tags"], list), f"{name} tags must be a list"
        assert isinstance(meta["difficulty"], list), (
            f"{name} difficulty must be a list"
        )


def test_find_scenarios_by_category(loader: DataLoader) -> None:
    results = loader.find_scenarios(category="restaurant")
    assert len(results) == 1
    assert results[0]["name"] == "restaurant"


def test_find_scenarios_by_tag(loader: DataLoader) -> None:
    results = loader.find_scenarios(tag="small-talk")
    # workplace and meeting-people both have small-talk
    names = {r["name"] for r in results}
    assert "workplace" in names
    assert "meeting-people" in names


def test_find_scenarios_by_difficulty(loader: DataLoader) -> None:
    advanced = loader.find_scenarios(difficulty="advanced")
    # workplace and phone-calls include advanced
    names = {r["name"] for r in advanced}
    assert "workplace" in names
    assert "phone-calls" in names

    beginner = loader.find_scenarios(difficulty="beginner")
    names = {r["name"] for r in beginner}
    assert "restaurant" in names
    assert "airport" in names
    assert "shopping" in names
    assert "meeting-people" in names


def test_find_scenarios_combined_filter(loader: DataLoader) -> None:
    # category=casual + difficulty=beginner
    results = loader.find_scenarios(difficulty="beginner")
    # Further narrow by tag
    filtered = [r for r in results if "ordering" in r["meta"].get("tags", [])]
    assert len(filtered) == 1
    assert filtered[0]["name"] == "restaurant"


def test_find_scenarios_no_match_returns_empty(loader: DataLoader) -> None:
    results = loader.find_scenarios(category="nonexistent")
    assert results == []


def test_find_vocabulary_by_level(loader: DataLoader) -> None:
    b1 = loader.find_vocabulary(level="B1")
    assert len(b1) == 1
    assert b1[0]["name"] == "b1-intermediate"


def test_find_vocabulary_by_tag(loader: DataLoader) -> None:
    results = loader.find_vocabulary(tag="small-talk")
    names = {r["name"] for r in results}
    assert "b1-intermediate" in names


def test_get_all_categories(loader: DataLoader) -> None:
    cats = loader.get_all_categories()
    assert set(cats) == {
        "restaurant",
        "airport",
        "workplace",
        "shopping",
        "meeting-people",
        "phone-calls",
    }
    assert cats == sorted(cats)  # sorted


def test_get_all_tags(loader: DataLoader) -> None:
    tags = loader.get_all_tags()
    assert "ordering" in tags
    assert "small-talk" in tags
    assert tags == sorted(tags)


def test_get_all_levels(loader: DataLoader) -> None:
    levels = loader.get_all_levels()
    assert set(levels) == {"A2", "B1", "B2"}


def test_scenario_content_includes_examples(loader: DataLoader) -> None:
    """Each scenario should have at least one example dialogue or common mistakes."""
    for name, data in loader.scenarios.items():
        content = data["content"].lower()
        assert any(
            marker in content
            for marker in ["example dialogue", "common mistakes", "key phrases"]
        ), f"{name} missing usable practice sections"
