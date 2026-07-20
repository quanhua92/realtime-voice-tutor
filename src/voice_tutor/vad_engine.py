"""Silero VAD with echo-aware dynamic thresholds.

The server instantiates one VADEngine per session (VAD has internal LSTM
state that must not bleed between users). All `process_chunk` calls are
synchronous CPU work; the server should call them via `asyncio.to_thread`
to keep the event loop responsive for barge-in cancellation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import (
    VAD_BARGE_IN_THRESHOLD,
    VAD_SAMPLE_RATE,
    VAD_SILENCE_TIMEOUT_MS,
    VAD_THRESHOLD,
)


class VADEngine:
    """Silero VAD with echo-aware dynamic thresholds.

    Echo-aware means: when the agent is speaking through the user's speakers,
    we raise the VAD threshold so microphone bleed from the speakers doesn't
    trigger a false barge-in. The user must speak *noticeably* louder than
    the agent echo to interrupt.
    """

    #: Silero accepts chunks of 256, 512, or 768 samples at 16kHz.
    CHUNK_SAMPLES = 512

    def __init__(self, sample_rate: int = VAD_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.model = load_silero_vad(onnx=True)

        # Per-instance speech-tracking state (mutated by process_chunk)
        self.silence_frames = 0
        chunk_ms = self.CHUNK_SAMPLES / sample_rate * 1000
        self.silence_timeout_frames = max(
            1, int(VAD_SILENCE_TIMEOUT_MS / chunk_ms)
        )
        self.is_speaking = False

    def reset(self) -> None:
        """Reset VAD model + speech-tracking state. Call on barge-in."""
        self.model.reset_states()
        self.silence_frames = 0
        self.is_speaking = False

    def process_chunk(
        self,
        pcm_bytes: bytes,
        agent_is_speaking: bool = False,
    ) -> dict[str, Any]:
        """Process a 512-sample 16kHz int16 PCM chunk.

        Args:
            pcm_bytes: raw little-endian int16 PCM bytes (512 samples = 1024 bytes).
            agent_is_speaking: when True, use the raised barge-in threshold
                so echo bleed doesn't trigger a false interruption.

        Returns:
            Dict with:
              - speech_prob: float in [0, 1]
              - is_speech: bool — chunk is above the *active* threshold
              - speech_ended: bool — silence timeout reached after speech
        """
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)

        speech_prob = float(self.model(tensor, self.sample_rate).item())

        threshold = VAD_BARGE_IN_THRESHOLD if agent_is_speaking else VAD_THRESHOLD
        is_speech = speech_prob > threshold

        speech_ended = False
        if is_speech:
            self.is_speaking = True
            self.silence_frames = 0
        elif self.is_speaking:
            self.silence_frames += 1
            if self.silence_frames >= self.silence_timeout_frames:
                speech_ended = True
                self.is_speaking = False
                self.silence_frames = 0

        return {
            "speech_prob": speech_prob,
            "is_speech": is_speech,
            "speech_ended": speech_ended,
        }
