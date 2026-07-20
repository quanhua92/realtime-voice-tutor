"""Tests for mcp_tools."""

from __future__ import annotations

from voice_tutor import mcp_tools
from voice_tutor.mcp_tools import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    check_vocabulary,
    get_scenario,
    lookup_phrases,
    suggest_topic,
)


# --- Registry / schema sanity ---

def test_tool_registry_has_four_tools() -> None:
    assert set(TOOL_REGISTRY.keys()) == {
        "get_scenario",
        "lookup_phrases",
        "check_vocabulary",
        "suggest_topic",
    }


def test_tool_schemas_match_registry() -> None:
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_REGISTRY.keys())


def test_each_schema_is_valid_openai_format() -> None:
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


# --- Tool behaviors ---

def test_get_scenario_no_args_returns_a_scenario() -> None:
    result = get_scenario()
    # Result is "**Title**\n\n# ... content"
    assert result.startswith("**")
    assert len(result) > 200
    # Should include key phrases or example dialogue
    assert "Key Phrases" in result or "Example Dialogue" in result


def test_get_scenario_by_category() -> None:
    result = get_scenario(category="restaurant")
    assert "Restaurant" in result
    assert "Ordering Coffee" in result


def test_get_scenario_by_difficulty_beginner() -> None:
    result = get_scenario(difficulty="beginner")
    # Multiple scenarios qualify; just verify we got one with content
    assert "**" in result  # bold title wrapper
    assert len(result) > 200


def test_get_scenario_invalid_category_falls_back() -> None:
    result = get_scenario(category="nonexistent")
    # Should fall back to any scenario rather than erroring
    assert len(result) > 100


def test_lookup_phrases_known_category() -> None:
    result = lookup_phrases(category="restaurant")
    assert "Ordering" in result or "Key Phrases" in result


def test_lookup_phrases_by_tag() -> None:
    result = lookup_phrases(category="small-talk")
    # 'small-talk' tag matches both workplace and meeting-people — the loader
    # returns the first match (workplace, alphabetically). Either way, the
    # result should be a populated scenario body.
    assert len(result) > 200
    assert "Key Phrases" in result or "Common Mistakes" in result


def test_lookup_phrases_unknown_category() -> None:
    result = lookup_phrases(category="nonexistent-tag-xyz")
    assert "No phrases found" in result
    # Should list available categories
    assert "restaurant" in result


def test_check_vocabulary_known_word() -> None:
    # 'commute' is in b1-intermediate
    result = check_vocabulary(word="commute")
    assert "[Level:" in result
    assert "commute" in result.lower()


def test_check_vocabulary_case_insensitive() -> None:
    result_lower = check_vocabulary(word="commute")
    result_upper = check_vocabulary(word="COMMUTE")
    assert "[Level:" in result_lower
    assert "[Level:" in result_upper


def test_check_vocabulary_unknown_word() -> None:
    result = check_vocabulary(word="xylophone")
    assert "not found" in result.lower()


def test_suggest_topic_returns_categories() -> None:
    result = suggest_topic()
    assert "How about practicing" in result
    assert "restaurant" in result  # at least one category listed
    assert "Available topics:" in result


def test_suggest_topic_no_data_returns_graceful(monkeypatch) -> None:
    """If no scenarios loaded, suggest_topic handles gracefully."""
    monkeypatch.setattr(mcp_tools.data, "get_all_categories", lambda: [])
    monkeypatch.setattr(mcp_tools.data, "get_all_tags", lambda: [])
    result = suggest_topic()
    assert "not sure" in result.lower() or "chat" in result.lower()
