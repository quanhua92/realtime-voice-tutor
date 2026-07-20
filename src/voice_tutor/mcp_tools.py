"""MCP-style tool definitions.

Each tool is a plain Python function that returns a string. Tool
schemas follow the OpenAI function-calling format and are passed to
the LLM via the `tools` field of `/v1/chat/completions`.

The LLM engine executes tools via the TOOL_REGISTRY lookup; results
are returned to the model as `{"role": "tool", "tool_call_id": ...,
"content": ...}` messages (OpenAI standard).
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from .data_loader import DataLoader

# Global data store — loaded once at module import.
data = DataLoader("data")


def get_scenario(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> str:
    """
    Find a conversation practice scenario.

    Args:
        category: scenario category (restaurant, airport, workplace,
            shopping, meeting-people, phone-calls). Optional.
        difficulty: beginner / intermediate / advanced. Optional.

    Returns:
        Scenario content with role-play setup, key phrases, and examples.
        If multiple scenarios match, one is chosen at random.
    """
    results = data.find_scenarios(category=category, difficulty=difficulty)
    if not results:
        # Fallback: any scenario
        results = data.find_scenarios()
    if not results:
        return "No scenarios available. Let's just have a free conversation!"

    chosen = random.choice(results)
    title = chosen["meta"].get("title", chosen["name"])
    return f"**{title}**\n\n{chosen['content']}"


def lookup_phrases(category: str) -> str:
    """
    Look up useful English phrases for a specific conversation category.

    Args:
        category: conversation category (e.g. 'restaurant', 'workplace')
            or a tag (e.g. 'small-talk', 'ordering').

    Returns:
        Key phrases and common mistakes for the matched scenario.
    """
    results = data.find_scenarios(category=category)
    if not results:
        results = data.find_scenarios(tag=category)
    if not results:
        available = ", ".join(data.get_all_categories())
        return f"No phrases found for '{category}'. Try: {available}"

    return results[0]["content"]


def check_vocabulary(word: str) -> str:
    """
    Look up an English word or phrase — definition, example usage, CEFR level.

    Args:
        word: the English word or phrase to look up.

    Returns:
        Matching vocabulary entries with level annotation, or a not-found note.
    """
    needle = word.lower().strip()
    matches = []
    for vocab in data.vocabulary.values():
        if needle in vocab["content"].lower():
            level = vocab["meta"].get("level", "unknown")
            matches.append((level, vocab["content"]))

    if not matches:
        return (
            f"'{word}' not found in vocabulary database. "
            "Try explaining it in context instead."
        )

    # Sort matches: prefer lower levels (more common words) first
    level_order = {"A2": 0, "B1": 1, "B2": 2, "C1": 3, "C2": 4}
    matches.sort(key=lambda m: level_order.get(m[0], 99))

    level, content = matches[0]
    return f"[Level: {level}]\n\n{content}"


def suggest_topic() -> str:
    """
    Suggest a random conversation topic to practice.

    Returns:
        A suggested category with the full list of available topics and tags.
    """
    categories = data.get_all_categories()
    tags = data.get_all_tags()
    if not categories:
        return "I'm not sure what topics to suggest — let's just chat!"
    category = random.choice(categories)
    tag_sample = ", ".join(tags[:8]) if tags else "none"
    return (
        f"How about practicing **{category}** conversations?\n"
        f"Available topics: {', '.join(categories)}\n"
        f"Sample tags: {tag_sample}"
    )


# --- Tool registry for runtime dispatch ---
TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "get_scenario": get_scenario,
    "lookup_phrases": lookup_phrases,
    "check_vocabulary": check_vocabulary,
    "suggest_topic": suggest_topic,
}


# --- OpenAI-format tool schemas (passed via `tools=` field) ---
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_scenario",
            "description": (
                "Find a conversation practice scenario by category or difficulty. "
                "Use when the user wants to practice a specific situation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Scenario category: restaurant, airport, workplace, "
                            "shopping, meeting-people, phone-calls"
                        ),
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "description": "Difficulty level.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_phrases",
            "description": (
                "Look up useful English phrases for a conversation category. "
                "Use when the user is stuck or asks how to say something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Conversation category (e.g. 'restaurant') "
                            "or tag (e.g. 'small-talk')."
                        ),
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_vocabulary",
            "description": (
                "Look up an English word or phrase — definition, examples, "
                "and CEFR level. Use when the user asks about a word or "
                "uses one incorrectly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The English word or phrase to look up.",
                    },
                },
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_topic",
            "description": (
                "Suggest a random conversation topic to practice. "
                "Use when the user doesn't know what to practice."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
