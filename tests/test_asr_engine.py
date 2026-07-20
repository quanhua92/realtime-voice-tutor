"""Tests for asr_engine.ASREngine.

Uses the Kokoro-generated speech fixture (already in English) to verify
transcription actually works. Skips if the fixture or model is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_tutor.asr_engine import ASREngine


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def asr() -> ASREngine:
    """Module-scoped — model load is several seconds."""
    return ASREngine()


@pytest.fixture(scope="module")
def speech_pcm_bytes() -> bytes:
    speech_path = FIXTURES_DIR / "hello_speech_16k.npy"
    if not speech_path.exists():
        pytest.skip(
            f"Speech fixture missing at {speech_path}. "
            "Run scripts/generate_test_fixtures.py first."
        )
    audio = np.load(speech_path)
    return audio.tobytes()


# --- Transcription correctness ---

def test_transcribes_known_english_speech(asr: ASREngine, speech_pcm_bytes: bytes) -> None:
    """The fixture says 'Hello there! This is a test of the voice activity
    detection...' — transcription should at minimum contain 'hello' or 'test'.
    """
    text = asr.transcribe(speech_pcm_bytes)
    assert isinstance(text, str)
    assert len(text) > 0
    text_lower = text.lower()
    # Whisper tiny.en is approximate; accept any of these key words
    acceptable = any(
        word in text_lower
        for word in ["hello", "test", "voice", "speaking", "detection"]
    )
    assert acceptable, f"Transcription {text!r} didn't contain expected words"


def test_transcription_is_stripped(asr: ASREngine, speech_pcm_bytes: bytes) -> None:
    text = asr.transcribe(speech_pcm_bytes)
    assert text == text.strip()


# --- Edge cases ---

def test_returns_empty_for_too_short_audio(asr: ASREngine) -> None:
    """Audio below MIN_AUDIO_BYTES (100ms = 3200 bytes) returns empty."""
    short = np.zeros(100, dtype=np.int16).tobytes()  # 50 samples
    assert asr.transcribe(short) == ""


def test_returns_empty_for_silence(asr: ASREngine) -> None:
    """One second of pure silence should not produce meaningful text."""
    silence = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second
    text = asr.transcribe(silence)
    # Whisper with vad_filter=True typically returns empty or ""
    assert isinstance(text, str)
    assert len(text) < 30  # very short or empty


def test_handles_partial_chunk(asr: ASREngine) -> None:
    """Just above the MIN_AUDIO_BYTES threshold — should not crash."""
    just_enough = np.zeros(3300, dtype=np.int16).tobytes()  # ~103ms
    text = asr.transcribe(just_enough)
    assert isinstance(text, str)


# --- Config / construction ---

def test_min_audio_bytes_constant() -> None:
    """MIN_AUDIO_BYTES should equal 100ms of 16kHz int16 audio."""
    expected = 16000 * 2 * 100 // 1000  # 3200 bytes
    assert ASREngine.MIN_AUDIO_BYTES == expected
