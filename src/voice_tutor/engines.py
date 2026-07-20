"""Shared singleton engine instances — loaded once at startup.

VAD is per-session (stateful LSTM context), so it's NOT shared.
ASR, LLM, and TTS are stateless and shared across all WebSocket sessions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

logger = logging.getLogger("voicetutor.engines")

# These get populated by the lifespan handler below.
# Typed as Optional so callers know to check; FastAPI startup guarantees
# they're set before any request is served.
asr_engine: Optional["ASREngine"] = None  # type: ignore[name-defined]
llm_engine: Optional["LLMEngine"] = None  # type: ignore[name-defined]
tts_engine: Optional["TTSEngine"] = None  # type: ignore[name-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — loads all stateless engines once at startup."""
    global asr_engine, llm_engine, tts_engine

    # Import here so module import doesn't trigger model loads (cleaner tests)
    from .asr_engine import ASREngine
    from .llm_engine import LLMEngine
    from .tts_engine import TTSEngine

    logger.info("Loading shared engines at startup...")

    logger.info("  → ASR (faster-whisper)...")
    asr_engine = ASREngine()

    logger.info("  → LLM (OpenAI-compat client)...")
    llm_engine = LLMEngine()

    logger.info("  → TTS (Kokoro)...")
    tts_engine = TTSEngine()

    logger.info("✅ All shared engines loaded.")

    try:
        yield
    finally:
        if llm_engine is not None:
            await llm_engine.close()
            logger.info("LLM client closed.")
