"""Faster-Whisper ASR with CTranslate2 INT8 quantization.

Loaded once at startup as a shared singleton (stateless). All
transcribe() calls are synchronous CPU work; the server wraps them in
asyncio.to_thread so barge-in task.cancel() propagates fast.
"""

from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from .config import ASR_COMPUTE_TYPE, ASR_DEVICE, ASR_MODEL_SIZE

logger = logging.getLogger("voicetutor.asr")


class ASREngine:
    """Faster-Whisper ASR with CTranslate2 INT8 quantization."""

    #: Minimum audio length (in bytes) to attempt transcription.
    #: Below this we return empty (avoid noisy transcriptions of <100ms).
    MIN_AUDIO_BYTES = 16000 * 2 * 100 // 1000  # 100ms @ 16kHz int16

    def __init__(
        self,
        model_size: str = ASR_MODEL_SIZE,
        device: str = ASR_DEVICE,
        compute_type: str = ASR_COMPUTE_TYPE,
    ) -> None:
        logger.info(
            f"Loading faster-whisper model={model_size} device={device} "
            f"compute_type={compute_type}"
        )
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("ASR model loaded")

    def transcribe(self, pcm_buffer: bytes) -> str:
        """Transcribe a raw int16 16kHz PCM byte buffer to text.

        Args:
            pcm_buffer: little-endian int16 16kHz mono PCM bytes.

        Returns:
            Transcribed text (stripped). Empty string if audio is too short
            or no speech was detected.
        """
        if len(pcm_buffer) < self.MIN_AUDIO_BYTES:
            return ""

        audio_int16 = np.frombuffer(pcm_buffer, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        segments, _info = self.model.transcribe(
            audio_float32,
            beam_size=1,
            language="en",
            vad_filter=True,  # skip internal silence segments
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text
