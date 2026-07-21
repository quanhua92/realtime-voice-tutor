"""Local Kokoro TTS engine with per-sentence streaming.

Loaded once at startup as a shared singleton (stateless synthesis).
All blocking synthesis calls are wrapped in asyncio.to_thread so
barge-in task.cancel() propagates within one event-loop tick.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import AsyncIterator

import numpy as np
from kokoro_onnx import Kokoro

from .config import (
    KOKORO_MODEL_PATH,
    KOKORO_VOICES_PATH,
    TTS_LANG,
    TTS_SPEED,
    TTS_VOICE,
)

logger = logging.getLogger("voicetutor.tts")


# ────────────────────────────────────────────────────────────────────
# TTS text sanitization
# ────────────────────────────────────────────────────────────────────
# Strip markdown / emoji / fake tool-call syntax before synthesis.

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F"
    "\U00002B00-\U00002BFF\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

_MD_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Fake tool-call syntax that small models emit as text
    (re.compile(r"<scenario>.*?</scenario>", re.DOTALL | re.IGNORECASE), " "),
    (re.compile(r"</?(?:scenario|tool_call|function|response)>", re.IGNORECASE), " "),
    (re.compile(r'\{"name"\s*:\s*"[^"]+".*?\}', re.DOTALL), " "),
    # Fenced code blocks
    (re.compile(r"```[a-zA-Z]*\n.*?\n```", re.DOTALL), " "),
    # Inline code
    (re.compile(r"`([^`]+)`"), r"\1"),
    # Bold+italic combos
    (re.compile(r"\*\*\*([^*]+)\*\*\*"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"(?<=\s)\*([^*\n]+)\*(?=\s)"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"~~([^~]+)~~"), r"\1"),
    # Headings
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # Bullets / numbered lists
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),
    # Blockquotes
    (re.compile(r"^\s*>\s*", re.MULTILINE), ""),
    # Horizontal rules
    (re.compile(r"^[\-\*_]{3,}$", re.MULTILINE), ""),
    # Links / images
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"!\[[^\]]*\]\([^)]+\)"), ""),
    # Standalone markers
    (re.compile(r"[→✗✓✔✘❌✅]"), " "),
    # Standalone label patterns like "**Feedback**:" or "**Tool used**:"
    (re.compile(r"\*?\*?(?:Feedback|Tool used|Note|Tip|Hint)\*?\*?\s*:", re.IGNORECASE), ""),
    # Multiple spaces / dashes
    (re.compile(r"\s+[-—]\s+"), ". "),
    (re.compile(r"[ \t]{2,}"), " "),
]


def sanitize_for_tts(text: str) -> str:
    """Strip markdown, emoji, fake tool-call syntax before synthesis."""
    if not text:
        return ""
    out = _EMOJI_RE.sub("", text)
    for pattern, replacement in _MD_PATTERNS:
        out = pattern.sub(replacement, out)
    out = re.sub(r"\s*\n\s*", ". ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\.\s*\.\s*\.\s*", "... ", out)
    return out.strip()


class TTSEngine:
    """Kokoro ONNX local text-to-speech engine.

    Always outputs 24kHz PCM. The browser UI upsamples to its AudioContext's
    native rate for playback.
    """

    SAMPLE_RATE = 24000

    def __init__(self) -> None:
        logger.info(
            f"Loading Kokoro TTS: model={KOKORO_MODEL_PATH} voices={KOKORO_VOICES_PATH}"
        )
        self.kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
        self.voice = TTS_VOICE
        self.speed = TTS_SPEED
        self.lang = TTS_LANG
        logger.info(
            f"Kokoro TTS loaded (voice={self.voice} speed={self.speed} lang={self.lang})"
        )

    def _synthesize_blocking(self, text: str) -> tuple[bytes, int]:
        """Synchronous synthesis — call via asyncio.to_thread."""
        clean = sanitize_for_tts(text)
        if not clean:
            return b"", self.SAMPLE_RATE

        samples, sr = self.kokoro.create(
            clean, voice=self.voice, speed=self.speed, lang=self.lang
        )
        audio_int16 = (np.asarray(samples) * 32767).astype(np.int16)
        return audio_int16.tobytes(), sr

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesize text → raw int16 PCM bytes.

        Returns (pcm_bytes, sample_rate). Empty input returns (b"", sample_rate).
        """
        return await asyncio.to_thread(self._synthesize_blocking, text)

    async def synthesize_stream(
        self, text: str
    ) -> AsyncIterator[tuple[bytes, int]]:
        """Stream synthesis in Kokoro's native chunks.

        Yields (pcm_bytes, sample_rate) tuples as Kokoro produces them.
        First chunk arrives sooner than full-sentence synthesis, lowering
        perceived TTFA on long sentences.

        Empty input yields nothing.
        """
        clean = sanitize_for_tts(text)
        if not clean:
            return

        async for samples, sr in self.kokoro.create_stream(
            clean, voice=self.voice, speed=self.speed, lang=self.lang
        ):
            audio_int16 = (np.asarray(samples) * 32767).astype(np.int16)
            yield audio_int16.tobytes(), sr

    async def synthesize_filler(self) -> tuple[bytes, int]:
        """Generate a short filler audio for tool-execution delays."""
        fillers = [
            "Let me check on that.",
            "One moment.",
            "Looking that up for you.",
        ]
        return await self.synthesize(random.choice(fillers))
