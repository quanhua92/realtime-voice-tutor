"""Generate test fixtures using Kokoro TTS.

Synthesizes a short speech sample with Kokoro, resamples 24kHz→16kHz,
and saves as `tests/fixtures/hello_speech_16k.npy` (numpy int16 array).

This is run manually (slow, requires Kokoro). The resulting fixture is
committed so unit tests stay fast and Kokoro-free:

    uv run python scripts/download_models.py
    uv run python scripts/generate_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

from voice_tutor.config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample — fine for VAD test fixtures."""
    if src_sr == dst_sr:
        return audio
    duration = len(audio) / src_sr
    new_len = int(round(duration * dst_sr))
    src_idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(src_idx, np.arange(len(audio)), audio)


def main() -> int:
    import sys
    if not Path(KOKORO_MODEL_PATH).exists():
        print(f"❌ Kokoro model not found at {KOKORO_MODEL_PATH}", file=sys.stderr)
        print("   Run `uv run python scripts/download_models.py` first.")
        return 1
    if not Path(KOKORO_VOICES_PATH).exists():
        print(f"❌ Kokoro voices not found at {KOKORO_VOICES_PATH}", file=sys.stderr)
        return 1

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

    # Speech sample — long enough to clearly trigger VAD (>1 sec @ 16kHz)
    text = (
        "Hello there! This is a test of the voice activity detection. "
        "I am speaking clearly so the model can recognize my voice."
    )
    print(f"🔊 Synthesizing: {text!r}")
    samples_24k, sr_24k = kokoro.create(
        text, voice="af_sarah", speed=1.0, lang="en-us"
    )
    print(f"   Got {len(samples_24k)} samples @ {sr_24k}Hz "
          f"({len(samples_24k)/sr_24k:.2f}s)")

    # Resample 24kHz → 16kHz
    samples_np = np.asarray(samples_24k, dtype=np.float32)
    samples_16k = _resample_linear(samples_np, sr_24k, 16000)
    print(f"   Resampled to {len(samples_16k)} samples @ 16000Hz "
          f"({len(samples_16k)/16000:.2f}s)")

    # Convert to int16
    audio_int16 = (np.clip(samples_16k, -1.0, 1.0) * 32767).astype(np.int16)

    # Save as numpy fixture
    speech_path = FIXTURES_DIR / "hello_speech_16k.npy"
    np.save(speech_path, audio_int16)
    print(f"💾 Saved speech fixture: {speech_path} ({audio_int16.nbytes} bytes)")

    # Also save a silence fixture of the same length
    silence = np.zeros_like(audio_int16)
    silence_path = FIXTURES_DIR / "silence_16k.npy"
    np.save(silence_path, silence)
    print(f"💾 Saved silence fixture: {silence_path} ({silence.nbytes} bytes)")

    print("\n✅ Test fixtures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
