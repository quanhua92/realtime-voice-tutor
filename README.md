# realtime-voice-tutor

Fully-local, sub-500ms TTFA real-time voice agent for casual English conversation practice.

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

# 3. Start Ollama (or any OpenAI-compatible provider)
ollama pull gemma4:cloud   # or: llama3.2:3b, qwen3:0.6b, etc.
ollama serve

# 4. Configure env (edit if needed — never overwrite existing .env)
cp .env.example .env

# 5. Run the server
uv run python -m voice_tutor.server
```

Open `http://localhost:8888` in Chrome. For LAN access or to use `0.0.0.0` / a LAN IP, you need HTTPS — see **Local HTTPS via Caddy** below.

## Local HTTPS via Caddy (required for non-localhost access)

Chrome's `getUserMedia` (microphone) requires a **secure context**. `http://localhost` counts as secure, but `http://<lan-ip>` or `http://0.0.0.0` does not — `navigator.mediaDevices` is undefined there. The fix is a local HTTPS proxy via Caddy + mkcert.

### One-time setup

```bash
# 1. Install Caddy and mkcert
brew install caddy mkcert

# 2. Install mkcert's root CA into the macOS keychain (prompts for sudo)
mkcert -install

# 3. Generate a cert with SANs for all the hostnames you'll use
LAN_IP=$(ipconfig getifaddr en0)        # e.g. 192.168.1.18
mkcert localhost 127.0.0.1 ::1 "$LAN_IP"

# 4. Move the certs to the location referenced by Caddyfile
mv localhost+3.pem static/tls-cert.pem
mv localhost+3-key.pem static/tls-key.pem
```

The certs are gitignored — never committed. If your LAN IP changes, regenerate.

### Run

```bash
# Terminal 1: backend (HTTP on :8888)
uv run python -m voice_tutor.server

# Terminal 2: Caddy proxy (HTTPS on :8443 → :8888)
caddy run --config Caddyfile
```

Open in Chrome:
- `https://localhost:8443` — same machine
- `https://192.168.1.18:8443` — LAN access (replace with your actual LAN IP)

If you see a cert warning after following the steps above, your browser didn't pick up the mkcert CA — retry `mkcert -install` or restart Chrome.

### Gotchas

- **`ERR_SSL_PROTOCOL_ERROR`**: Chrome caches HSTS / Alt-Svc headers. Open `chrome://net-internals/#hsts` → "Delete domain security policies" → enter the hostname. The Caddyfile disables HTTP/3 (`servers { protocols h1 h2 }`) for this reason.
- **Mixed-content WS error**: page is HTTPS but WebSocket is `ws://`. The UI already auto-switches to `wss://` based on `location.protocol`.
- **`0.0.0.0` is non-routable**: Chrome converts it to `127.0.0.1` internally, which breaks SNI matching. Use `localhost` or your LAN IP instead.
- **Cert SANs**: the cert must list every hostname you'll access via. Regenerate with `mkcert localhost 127.0.0.1 ::1 <new-ip>` if your setup changes.

## Latency targets

| Stage | Target |
|---|---|
| TTFA (speech-end → audio) | ~370ms |
| Perceived (incl. 500ms endpointing) | ~870ms |
| Barge-in flush | <50ms |

## Tests

```bash
# Fast unit tests (no external services needed)
uv run pytest tests/ -m "not integration" -v

# Integration tests (require Ollama + Kokoro models downloaded)
uv run pytest tests/ -m integration -v
```

## Project structure

```
realtime-voice-tutor/
├── pyproject.toml              # uv project config & dependencies
├── Caddyfile                   # Local HTTPS proxy config
├── config (env-driven):        # .env.example template
├── src/voice_tutor/            # Package: all engine modules
│   ├── config.py               # Centralized env-driven config
│   ├── server.py               # FastAPI WebSocket orchestrator & barge-in
│   ├── engines.py              # Shared singleton registry (FastAPI lifespan)
│   ├── vad_engine.py           # Silero VAD with echo-aware thresholds
│   ├── asr_engine.py           # Faster-Whisper worker
│   ├── llm_engine.py           # OpenAI-compat streaming + tool loop
│   ├── tts_engine.py           # Kokoro local TTS
│   ├── mcp_tools.py            # 4 tool functions + OpenAI schemas
│   └── data_loader.py          # Markdown + frontmatter loader
├── data/                       # Scenarios + vocabulary (Markdown)
├── static/                     # Browser UI (HTML + AudioWorklet)
├── scripts/                    # download_models.py, generate_test_fixtures.py
├── tests/                      # pytest unit + integration tests
└── docs/                       # Design notes (PLAN.md, COMMIT_PLAN.md)
```

## Limitations (POC)

- No WebSocket auth
- No conversation persistence
- English-only
- Single-user load-testing
