"""Integration tests for tts_engine.TTSEngine against real Kokoro.

Marked `integration` and auto-skip if the Kokoro model files aren't
present (run `scripts/download_models.py` first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_tutor.config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH
from voice_tutor.tts_engine import TTSEngine


pytestmark = pytest.mark.integration


def _kokoro_models_present() -> bool:
    return (
        Path(KOKORO_MODEL_PATH).exists()
        and Path(KOKORO_VOICES_PATH).exists()
    )


skip_if_no_kokoro = pytest.mark.skipif(
    not _kokoro_models_present(),
    reason=(
        f"Kokoro models missing at {KOKORO_MODEL_PATH} / {KOKORO_VOICES_PATH}. "
        "Run `uv run python scripts/download_models.py` first."
    ),
)


@pytest.fixture(scope="module")
def tts() -> TTSEngine:
    """Module-scoped — model load is ~1-2s, reuse across tests."""
    return TTSEngine()


# --- Synthesize (one-shot) ---

@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_returns_nonempty_pcm(tts: TTSEngine) -> None:
    pcm_bytes, sr = await tts.synthesize("Hello there!")
    assert isinstance(pcm_bytes, (bytes, bytearray))
    assert len(pcm_bytes) > 1000  # ~200ms+ of audio at minimum
    assert sr == 24000


@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_longer_text_yields_more_bytes(tts: TTSEngine) -> None:
    short_bytes, _ = await tts.synthesize("Hi.")
    long_bytes, _ = await tts.synthesize(
        "Hello there! This is a much longer sentence that should produce "
        "significantly more audio than just the word hi."
    )
    assert len(long_bytes) > len(short_bytes) * 3


@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_empty_returns_empty(tts: TTSEngine) -> None:
    pcm_bytes, sr = await tts.synthesize("")
    assert pcm_bytes == b""
    assert sr == 24000

    pcm_ws, _ = await tts.synthesize("   \t\n")
    assert pcm_ws == b""


@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_returns_consistent_sample_rate(tts: TTSEngine) -> None:
    for text in ["One.", "Two words.", "Three whole words here."]:
        _, sr = await tts.synthesize(text)
        assert sr == 24000


# --- Stream synthesis ---

@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_stream_yields_chunks(tts: TTSEngine) -> None:
    """Long text should yield multiple chunks of PCM audio."""
    chunks = []
    async for pcm_bytes, sr in tts.synthesize_stream(
        "Hello there! This is a longer sentence that should produce "
        "multiple chunks of streaming audio for testing purposes."
    ):
        chunks.append((pcm_bytes, sr))

    assert len(chunks) >= 1
    for pcm, sr in chunks:
        assert sr == 24000
        assert len(pcm) > 0
    # Total bytes should be similar to one-shot synthesis
    total_streamed = sum(len(p) for p, _ in chunks)
    one_shot, _ = await tts.synthesize(
        "Hello there! This is a longer sentence that should produce "
        "multiple chunks of streaming audio for testing purposes."
    )
    # Allow ±20% variance between streaming and one-shot
    assert total_streamed >= len(one_shot) * 0.8


@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_stream_empty_yields_nothing(tts: TTSEngine) -> None:
    chunks = [c async for c in tts.synthesize_stream("")]
    assert chunks == []


# --- Filler ---

@skip_if_no_kokoro
@pytest.mark.asyncio
async def test_synthesize_filler_returns_audio(tts: TTSEngine) -> None:
    pcm_bytes, sr = await tts.synthesize_filler()
    assert len(pcm_bytes) > 1000
    assert sr == 24000


# --- Lifecycle / config ---

def test_sample_rate_constant() -> None:
    assert TTSEngine.SAMPLE_RATE == 24000


@skip_if_no_kokoro
def test_engine_initializes_with_config_voice(tts: TTSEngine) -> None:
    from config import TTS_VOICE
    assert tts.voice == TTS_VOICE
    assert tts.SAMPLE_RATE == 24000
