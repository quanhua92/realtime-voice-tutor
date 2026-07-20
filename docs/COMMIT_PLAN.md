# Commit Plan — 11 Reviewable Commits

Each commit is independently runnable/testable. Linear history. No commit depends on a later one.

| # | Commit message | Key verification |
|---|---|---|
| 0 | `docs: revise plan with locked architecture decisions` | Diff review; PLAN.md self-consistent with Locked Decisions section |
| 1 | `chore: project skeleton with uv and deps` | `uv sync` works; `python -c "import fastapi, openai, faster_whisper, kokoro_onnx, silero_vad; print('ok')"` runs |
| 2 | `feat(config): centralized env-driven configuration` | `python config.py` prints resolved config |
| 3 | `feat(data): scenario and vocabulary content with loader` | `pytest tests/test_data_loader.py` |
| 4 | `feat(mcp-tools): function registry and OpenAI schemas` | `pytest tests/test_mcp_tools.py` |
| 5 | `feat(vad): Silero VAD with echo-aware thresholds` | `pytest tests/test_vad_engine.py` |
| 6 | `feat(asr): faster-whisper worker` | `pytest tests/test_asr_engine.py` |
| 7 | `feat(llm): OpenAI-compatible streaming client with tool loop` | `pytest tests/test_llm_engine.py` |
| 8 | `feat(tts): Kokoro streaming engine` | `pytest tests/test_tts_engine.py` |
| 9 | `feat(server): WebSocket orchestrator with 4-step barge-in` | integration test happy-path + barge-in |
| 10 | `feat(ui): browser UI with AudioWorklet capture and barge-in` | manual test in Chrome |

## Locked architecture decisions (summary)

See **Locked Decisions** section at top of `docs/PLAN.md` for the authoritative list. Highlights:

- **LLM**: OpenAI `/v1/chat/completions` on Ollama (swappable with Groq/NVIDIA NIM via env vars)
- **Tool format**: OpenAI standard (`tool_call_id`, JSON-string `arguments`)
- **Mic capture**: AudioWorklet at native rate → resample to 16kHz in worklet
- **TTS playback**: single AudioContext at native rate, per-chunk upsample 24kHz→native
- **VAD silence timeout**: 500ms (configurable)
- **All blocking work**: `asyncio.to_thread()` for preemption-friendly barge-in
- **Models**: shared singletons loaded once at startup (FastAPI `lifespan`)
- **TTS**: Kokoro v1.0 with per-sentence streaming
- **Silero VAD**: pip package (no torch dep, true offline)

## Latency targets

| Stage | Target |
|---|---|
| Mic + AEC + Worklet + resample | 15ms |
| VAD per chunk (off-thread) | 2ms |
| Endpointing silence (perceived, not in TTFA) | 500ms |
| ASR `tiny.en` INT8 | 90ms |
| LLM TTFT (llama3.2:3b) | 180ms |
| TTS first stream chunk | 60ms |
| Network + buffer | 20ms |
| **TTFA (speech-end → audio)** | **~370ms** |
| **Perceived (silence + TTFA)** | **~870ms** |
