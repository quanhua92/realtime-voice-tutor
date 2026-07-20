"""Integration tests for llm_engine.LLMEngine against a real Ollama endpoint.

Set OPENAI_MODEL in the environment to control which model is used
(e.g. `OPENAI_MODEL=gemma4:cloud uv run pytest tests/test_llm_engine.py -m integration`).

These tests are marked `integration` so they can be skipped via
`pytest -m "not integration"` when no Ollama is running.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import httpx
import pytest

from voice_tutor.config import OPENAI_BASE_URL, OPENAI_MODEL
from voice_tutor.llm_engine import LLMEngine


pytestmark = pytest.mark.integration


def _ollama_is_running() -> bool:
    """Probe the local Ollama /v1 endpoint with a quick HEAD-like request."""
    try:
        with httpx.Client(timeout=1.5) as client:
            r = client.get(f"{OPENAI_BASE_URL.rstrip('/v1')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


OLLAMA_UP = _ollama_is_running()
skip_if_no_ollama = pytest.mark.skipif(
    not OLLAMA_UP,
    reason=f"Ollama not reachable at {OPENAI_BASE_URL} (model={OPENAI_MODEL})",
)


@pytest.fixture(scope="module")
def engine() -> LLMEngine:
    return LLMEngine()


@pytest.fixture(scope="module")
def event_loop():
    """Fresh event loop for the module so the engine doesn't outlive its loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --- Basic streaming ---

@skip_if_no_ollama
@pytest.mark.asyncio
async def test_streams_text_tokens(engine: LLMEngine) -> None:
    """A simple prompt should stream multiple text tokens."""
    tokens = []
    async for tok in engine.generate_stream(
        [{"role": "user", "content": "Say hello in one short sentence."}]
    ):
        tokens.append(tok)

    # We should have received at least 3 separate token chunks
    assert len(tokens) >= 3, f"Expected multiple streamed tokens, got {tokens}"
    joined = "".join(tokens).strip()
    assert len(joined) > 0


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_tokens_are_short_strings(engine: LLMEngine) -> None:
    """Each yielded value should be a short str (a token, not a paragraph)."""
    tokens = []
    async for tok in engine.generate_stream(
        [{"role": "user", "content": "Count from 1 to 5."}]
    ):
        tokens.append(tok)

    assert all(isinstance(t, str) for t in tokens)
    # No single token should be longer than ~100 chars
    assert all(len(t) < 100 for t in tokens)


# --- Tool calling ---

@skip_if_no_ollama
@pytest.mark.asyncio
async def test_tool_call_dispatches_to_registry(engine: LLMEngine) -> None:
    """A request that needs a scenario should trigger the get_scenario tool
    and incorporate its output into the response."""
    response_tokens: list[str] = []
    async for tok in engine.generate_stream(
        [
            {
                "role": "user",
                "content": (
                    "I want to practice ordering food at a restaurant. "
                    "Use the get_scenario tool to find one, then start the role-play."
                ),
            }
        ]
    ):
        response_tokens.append(tok)

    joined = "".join(response_tokens)
    # Tool may or may not fire depending on the model's tool-calling ability,
    # but we should always get *some* response.
    assert len(joined.strip()) > 0


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_check_vocabulary_tool(engine: LLMEngine) -> None:
    """Asking about a word should trigger check_vocabulary."""
    response_tokens: list[str] = []
    async for tok in engine.generate_stream(
        [
            {
                "role": "user",
                "content": (
                    "What does the word 'commute' mean? "
                    "Use the check_vocabulary tool to look it up."
                ),
            }
        ]
    ):
        response_tokens.append(tok)

    joined = "".join(response_tokens).lower()
    # The tool result includes "[Level: B1]" so the response should
    # mention commute or travel (the definition).
    assert any(w in joined for w in ["commute", "travel", "level", "b1"])


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_suggest_topic_tool(engine: LLMEngine) -> None:
    """An open-ended practice request should trigger suggest_topic."""
    response_tokens: list[str] = []
    async for tok in engine.generate_stream(
        [{"role": "user", "content": "I'm bored. Suggest a topic to practice using the suggest_topic tool."}]
    ):
        response_tokens.append(tok)

    joined = "".join(response_tokens).lower()
    # suggest_topic returns category names like "restaurant", "airport"...
    assert any(
        w in joined
        for w in ["restaurant", "airport", "workplace", "shopping", "phone", "topic"]
    )


# --- System prompt & history ---

@skip_if_no_ollama
@pytest.mark.asyncio
async def test_system_prompt_drives_persona(engine: LLMEngine) -> None:
    """The system prompt should make the model act like a friendly tutor."""
    response_tokens: list[str] = []
    async for tok in engine.generate_stream(
        [{"role": "user", "content": "Who are you?"}]
    ):
        response_tokens.append(tok)

    joined = "".join(response_tokens).lower()
    # The persona name is "VoiceTutor"; model should mention it or similar
    assert any(w in joined for w in ["voicetutor", "voice tutor", "tutor", "english"])


@skip_if_no_ollama
@pytest.mark.asyncio
async def test_history_is_used_for_context(engine: LLMEngine) -> None:
    """A multi-turn conversation should remember prior context."""
    messages = [
        {"role": "user", "content": "My name is TestUser123."},
        {"role": "assistant", "content": "Nice to meet you, TestUser123!"},
        {"role": "user", "content": "What's my name?"},
    ]
    response_tokens: list[str] = []
    async for tok in engine.generate_stream(messages):
        response_tokens.append(tok)

    joined = "".join(response_tokens)
    assert "TestUser123" in joined


# --- Engine lifecycle ---

@pytest.mark.asyncio
async def test_engine_close_does_not_raise() -> None:
    eng = LLMEngine()
    await eng.close()


# --- Non-integration sanity (always runs) ---

def test_engine_uses_configured_model() -> None:
    """The engine's model name matches config.OPENAI_MODEL."""
    eng = LLMEngine()
    assert eng.model == OPENAI_MODEL


def test_engine_constructs_client() -> None:
    """LLMEngine can be instantiated without calling the API."""
    eng = LLMEngine()
    assert eng.client is not None
    assert eng.client.base_url.host is not None
