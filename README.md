# realtime-voice-tutor

Fully-local, sub-500ms TTFA real-time voice agent for casual English conversation practice.

> See [`docs/PLAN.md`](docs/PLAN.md) for the full architectural plan and [`docs/COMMIT_PLAN.md`](docs/COMMIT_PLAN.md) for the 11-commit execution roadmap.

## Features

- 🎙️ **Semantic VAD** with echo-aware dynamic thresholds (Silero ONNX)
- ⚡ **Barge-in interruption** with 4-step state reconstruction
- 📝 **Streaming ASR** (Faster-Whisper `tiny.en` INT8)
- 🤖 **Agentic LLM tool loop** via OpenAI-compatible endpoint (Ollama / Groq / NVIDIA NIM)
- 🔊 **Local Kokoro TTS** with per-sentence streaming
- 🎨 **AudioWorklet-based browser UI** with proper PCM resampling

## Quick start

```bash
# 1. Install deps
uv sync

# 2. Download model files (one-time)
uv run python scripts/download_models.py

# 3. Start Ollama
ollama pull llama3.2:3b
ollama serve

# 4. Configure env (edit if needed — never overwrite existing .env)
cp .env.example .env

# 5. Run
uv run python server.py
```

Open `http://localhost:8888` in Chrome.

## Latency targets

| Stage | Target |
|---|---|
| TTFA (speech-end → audio) | ~370ms |
| Perceived (incl. 500ms endpointing) | ~870ms |
| Barge-in flush | <50ms |

## Tests

```bash
uv run pytest tests/ -v
```

## Project structure

See [`docs/PLAN.md` → Project Structure](docs/PLAN.md#-project-structure).

## Limitations (POC)

- No WebSocket auth
- No conversation persistence
- English-only
- Single-user load-testing
