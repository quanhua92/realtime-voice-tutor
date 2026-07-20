"""Centralized configuration with env overrides.

All settings are loaded once at startup from environment variables (or `.env`)
with sensible defaults. Swapping Ollama for Groq / NVIDIA NIM / OpenAI only
requires changing `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. Real env vars still take precedence (load_dotenv default).
load_dotenv()

# Project root = 3 levels up from src/voice_tutor/config.py
#   config.py → voice_tutor/ → src/ → <project_root>
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"


def _get_bool(key: str, default: str) -> bool:
    return os.getenv(key, default).lower() in {"1", "true", "yes", "on"}


def _get_int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


def _get_float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


# --- LLM (OpenAI-compatible endpoint; works with Ollama, Groq, NVIDIA NIM) ---
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
# Local Ollama ignores the API key value but the SDK requires one to be set.
# For cloud providers, set this to your real API key in .env.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gemma4:cloud")
LLM_HISTORY_TURNS: int = _get_int("LLM_HISTORY_TURNS", "10")
LLM_MAX_TOOL_ROUNDS: int = _get_int("LLM_MAX_TOOL_ROUNDS", "3")

# --- ASR ---
ASR_MODEL_SIZE: str = os.getenv("ASR_MODEL_SIZE", "tiny.en")
ASR_DEVICE: str = os.getenv("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE: str = os.getenv("ASR_COMPUTE_TYPE", "int8")

# --- VAD ---
VAD_THRESHOLD: float = _get_float("VAD_THRESHOLD", "0.6")
VAD_BARGE_IN_THRESHOLD: float = _get_float("VAD_BARGE_IN_THRESHOLD", "0.92")
VAD_SILENCE_TIMEOUT_MS: int = _get_int("VAD_SILENCE_TIMEOUT_MS", "500")
VAD_SAMPLE_RATE: int = _get_int("VAD_SAMPLE_RATE", "16000")
VAD_CHUNK_SAMPLES: int = _get_int("VAD_CHUNK_SAMPLES", "512")
SILERO_MODEL_PATH: str = os.getenv(
    "SILERO_MODEL_PATH", str(MODELS_DIR / "silero_vad.onnx")
)

# --- TTS ---
TTS_VOICE: str = os.getenv("TTS_VOICE", "af_sarah")
TTS_SPEED: float = _get_float("TTS_SPEED", "1.0")
TTS_LANG: str = os.getenv("TTS_LANG", "en-us")
TTS_STREAM: bool = _get_bool("TTS_STREAM", "true")
KOKORO_MODEL_PATH: str = os.getenv(
    "KOKORO_MODEL_PATH", str(MODELS_DIR / "kokoro-v1.0.onnx")
)
KOKORO_VOICES_PATH: str = os.getenv(
    "KOKORO_VOICES_PATH", str(MODELS_DIR / "voices-v1.0.bin")
)

# --- Server ---
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = _get_int("PORT", "8888")
LOG_LATENCY: bool = _get_bool("LOG_LATENCY", "true")

# --- Audio plumbing ---
AUDIO_SAMPLE_RATE: int = VAD_SAMPLE_RATE  # 16kHz for mic → VAD/ASR path
AUDIO_BYTES_PER_SAMPLE: int = 2  # int16 PCM
PRE_SPEECH_MS: int = _get_int("PRE_SPEECH_MS", "200")  # rolling pre-speech buffer

# --- System Prompt ---
SYSTEM_PROMPT: str = """You are a friendly English conversation practice tutor called VoiceTutor.

Your role:
1. Help users practice casual English through natural role-play conversations.
2. Gently correct grammar and suggest more natural phrasing when appropriate.
3. Use available tools to find relevant practice scenarios and useful phrases.
4. Keep the conversation flowing naturally — don't be overly formal or lecture-like.
5. After a few exchanges in a scenario, provide brief feedback on what went well.
6. Adjust difficulty based on the user's apparent level.
7. Keep responses under 2-3 sentences to feel conversational.

When the user wants to practice, use get_scenario to find a relevant situation,
then role-play that scenario with them. Use lookup_phrases to help when they're stuck.
"""


def resolved_config() -> dict:
    """Return a dict of all resolved config values (for logging / debugging)."""
    return {
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "OPENAI_MODEL": OPENAI_MODEL,
        "LLM_HISTORY_TURNS": LLM_HISTORY_TURNS,
        "LLM_MAX_TOOL_ROUNDS": LLM_MAX_TOOL_ROUNDS,
        "ASR_MODEL_SIZE": ASR_MODEL_SIZE,
        "ASR_DEVICE": ASR_DEVICE,
        "ASR_COMPUTE_TYPE": ASR_COMPUTE_TYPE,
        "VAD_THRESHOLD": VAD_THRESHOLD,
        "VAD_BARGE_IN_THRESHOLD": VAD_BARGE_IN_THRESHOLD,
        "VAD_SILENCE_TIMEOUT_MS": VAD_SILENCE_TIMEOUT_MS,
        "VAD_SAMPLE_RATE": VAD_SAMPLE_RATE,
        "VAD_CHUNK_SAMPLES": VAD_CHUNK_SAMPLES,
        "TTS_VOICE": TTS_VOICE,
        "TTS_SPEED": TTS_SPEED,
        "TTS_LANG": TTS_LANG,
        "TTS_STREAM": TTS_STREAM,
        "KOKORO_MODEL_PATH": KOKORO_MODEL_PATH,
        "KOKORO_VOICES_PATH": KOKORO_VOICES_PATH,
        "SILERO_MODEL_PATH": SILERO_MODEL_PATH,
        "HOST": HOST,
        "PORT": PORT,
        "LOG_LATENCY": LOG_LATENCY,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(resolved_config(), indent=2))
