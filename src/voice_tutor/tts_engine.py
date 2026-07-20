"""Local Kokoro TTS engine with per-sentence streaming.

Loaded once at startup as a shared singleton (stateless synthesis).
All blocking synthesis calls are wrapped in asyncio.to_thread so
barge-in task.cancel() propagates within one event-loop tick.
"""

from __future__ import annotations

import asyncio
import logging
import random
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
        if not text.strip():
            return b"", self.SAMPLE_RATE

        samples, sr = self.kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
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
        if not text.strip():
            return

        # kokoro.create_stream is an async generator wrapping blocking work.
        # We trust its internal scheduling — Kokoro chunks are short enough
        # (~200ms each) that barge-in latency is acceptable.
        async for samples, sr in self.kokoro.create_stream(
            text, voice=self.voice, speed=self.speed, lang=self.lang
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
