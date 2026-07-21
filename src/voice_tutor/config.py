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
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "qwen3:0.6b")
LLM_HISTORY_TURNS: int = _get_int("LLM_HISTORY_TURNS", "10")
LLM_MAX_TOOL_ROUNDS: int = _get_int("LLM_MAX_TOOL_ROUNDS", "2")
# Small models like qwen3:0.6b hallucinate fake tool-call syntax as
# plain text instead of using the API. Default off — flip to true when
# using a stronger model (qwen3.5:2b, llama3.2:3b, etc).
TOOLS_ENABLED: bool = _get_bool("TOOLS_ENABLED", "false")

# --- ASR ---
ASR_MODEL_SIZE: str = os.getenv("ASR_MODEL_SIZE", str(MODELS_DIR / "asr" / "base.en"))
ASR_DEVICE: str = os.getenv("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE: str = os.getenv("ASR_COMPUTE_TYPE", "int8")

# --- VAD ---
VAD_THRESHOLD: float = _get_float("VAD_THRESHOLD", "0.6")
# Barge-in threshold: VAD prob must exceed this during agent speech to
# trigger interruption. 0.85 catches normal-volume speech (observed
# probs 0.85-0.99) while filtering background noise that occasionally
# spikes to ~0.82. Below 0.85 produced false barge-ins on noise.
VAD_BARGE_IN_THRESHOLD: float = _get_float("VAD_BARGE_IN_THRESHOLD", "0.85")
VAD_SILENCE_TIMEOUT_MS: int = _get_int("VAD_SILENCE_TIMEOUT_MS", "800")
# Grace period at the start of agent speech during which barge-in is
# disabled entirely. Without this, agent TTS bleed into the mic triggers
# false barge-ins within 500ms of the agent starting to speak — the user
# hasn't even heard the agent yet. 1200ms gives the agent time to finish
# its first sentence before interruption is allowed.
VAD_BARGE_IN_GRACE_MS: int = _get_int("VAD_BARGE_IN_GRACE_MS", "1200")
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
# Note: qwen3 family supports thinking mode. Ollama's /v1/ OpenAI-compat
# endpoint does NOT forward the `think: false` parameter, so we suppress
# thinking via the /no_think keyword in the prompt. This is essential for
# latency — without it, qwen3.5:2b generates ~600 reasoning tokens before
# answering, adding 2-4 seconds to every turn.
SYSTEM_PROMPT: str = """/no_think
You are a friendly English conversation practice partner. The user is practicing spoken English with you over voice. Your replies will be read aloud by a text-to-speech engine.

Rules:
1. ROLE-PLAY in character. Pick a situation (café, airport, office, party, store, phone call) based on the user's first message and BE a real person in that situation. Don't describe the scenario or ask "what would you like to practice" — just start playing your role.
2. Speak in 1-2 short sentences per turn. Use plain conversational English. The user needs time to talk too.
3. Ask follow-up questions to keep the conversation moving.
4. If the user makes a grammar mistake, reply naturally first, then offer a more natural phrasing in parentheses. Example: "Sure thing! (More natural: 'Could I get a coffee?')"
5. Never output markdown, emoji, asterisks, bullet points, dashes, or any formatting characters — they get read aloud and sound bad.
6. Never mention tools, scenarios, or topics as if they were features. Just have the conversation.
7. Stay in character. Never say "as an AI" or "let me help you practice."
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
