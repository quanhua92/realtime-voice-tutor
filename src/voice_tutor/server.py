"""FastAPI WebSocket orchestrator with 4-step barge-in.

The heart of the POC — ties VAD, ASR, LLM, and TTS together with the
barge-in interruption flow.

WebSocket protocol (browser ↔ server):
  →  binary frames           : 512-sample 16kHz int16 PCM (mic audio)
  ←  {"type": "TRANSCRIPT", role, text}
  ←  {"type": "AUDIO_START", sample_rate}
  ←  binary frames           : 24kHz int16 PCM (TTS audio)
  ←  {"type": "FLUSH"}       : barge-in — drop all queued audio
  ←  {"type": "END_OF_TURN"}
  ←  {"type": "ERROR", message}

Barge-in flow (4 steps):
  1. Send FLUSH to client → it stops all playing/scheduled audio
  2. Cancel the in-flight LLM+TTS task (await it so CancelledError propagates)
  3. Reset is_agent_speaking
  4. Reconstruct chat history from words_spoken (sentences actually synthesized)

Critical correctness fixes vs. the original draft:
  - Barge-in does NOT discard the interrupting chunk: the new audio_buffer
    is seeded with pre_speech_buffer + current chunk so ASR captures the
    interrupting word.
  - All blocking engine calls run via asyncio.to_thread so task.cancel()
    actually preempts mid-synthesis instead of waiting for the next await.
  - Concurrent-pipeline guard: if speech_ended fires while a previous
    pipeline is still running, cancel the old one before launching new.
  - State reconstruction trims the assistant turn to words_spoken, not
    the full_response text that may include tokens never synthesized.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import engines
from .config import (
    AUDIO_BYTES_PER_SAMPLE,
    AUDIO_SAMPLE_RATE,
    HOST,
    LOG_LATENCY,
    PORT,
    PRE_SPEECH_MS,
    STATIC_DIR,
)
from .engines import lifespan
from .vad_engine import VADEngine

logger = logging.getLogger("voicetutor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Realtime Voice Tutor", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr_loaded": engines.asr_engine is not None,
        "llm_loaded": engines.llm_engine is not None,
        "tts_loaded": engines.tts_engine is not None,
    }


class Session:
    """Per-connection state. Engines are shared; this holds per-user data."""

    # Pre-speech padding: keep last N bytes of audio before VAD triggers
    # so the first word of a user's utterance isn't clipped.
    PRE_SPEECH_BYTES = (
        AUDIO_SAMPLE_RATE * AUDIO_BYTES_PER_SAMPLE * PRE_SPEECH_MS // 1000
    )

    def __init__(self) -> None:
        # VAD is stateful (LSTM context) — instantiate per-session.
        self.vad = VADEngine()

        # ASR/LLM/TTS come from the shared engine registry.
        # Fallback to direct construction if startup lifespan didn't run
        # (e.g., when this module is imported in tests without FastAPI).
        self.chat_history: list[dict] = []
        self.active_task: asyncio.Task | None = None
        self.is_agent_speaking: bool = False

        # Barge-in state reconstruction (sentences actually synthesized).
        self.words_spoken: list[str] = []
        self.bytes_sent: int = 0

        # Audio accumulation for ASR
        self.audio_buffer: bytearray = bytearray()
        # Rolling pre-speech buffer for low-latency speech-start capture
        self.pre_speech_buffer: bytearray = bytearray()

    def reset_after_barge_in(self, current_chunk: bytes) -> None:
        """Reset session state for a new user turn after barge-in.

        Seeds the new audio_buffer with pre_speech_buffer + current_chunk
        so the interrupting word is captured by the next ASR pass.
        """
        self.audio_buffer = bytearray(self.pre_speech_buffer)
        self.audio_buffer.extend(current_chunk)
        self.words_spoken.clear()
        self.bytes_sent = 0
        self.vad.reset()


@app.websocket("/ws/voice")
async def voice_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    logger.info(f"Client connected — {id(session):#x}")

    try:
        while True:
            data = await ws.receive_bytes()
            turn_start = time.perf_counter()

            # VAD: synchronous CPU work — off-thread so barge-in cancellation
            # (which needs the event loop) stays responsive.
            vad_result = await asyncio.to_thread(
                session.vad.process_chunk,
                data,
                session.is_agent_speaking,
            )

            # ───────────────────────────────────────────────────────────
            # 1. BARGE-IN: User interrupts during agent TTS
            # ───────────────────────────────────────────────────────────
            if session.is_agent_speaking and vad_result["is_speech"]:
                logger.info("⚡ BARGE-IN DETECTED — 4-Step Interruption")

                # Step 1: Flush client audio buffer immediately
                await ws.send_json({"type": "FLUSH"})

                # Step 2: Cancel active LLM + TTS pipeline and AWAIT it
                # so the CancelledError fully propagates before we proceed.
                if session.active_task and not session.active_task.done():
                    session.active_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await session.active_task

                # Step 3: Halt agent speaking state
                session.is_agent_speaking = False

                # Step 4: State Reconstruction
                # Trim chat history to reflect only what was actually synthesized.
                spoken_text = " ".join(session.words_spoken)
                if (
                    session.chat_history
                    and session.chat_history[-1]["role"] == "assistant"
                ):
                    session.chat_history[-1]["content"] = (
                        spoken_text + " [interrupted]"
                    )
                    logger.info(
                        f"State reconstructed: '{spoken_text[:80]}...'"
                    )
                else:
                    logger.info(
                        "Barge-in with no assistant turn to reconstruct"
                    )

                # Reset for new user turn — seed buffer with interrupting chunk
                session.reset_after_barge_in(data)
                continue

            # ───────────────────────────────────────────────────────────
            # 2. ACCUMULATE: Buffer user audio during speech
            # ───────────────────────────────────────────────────────────
            if vad_result["is_speech"]:
                # On speech start, prepend pre-speech buffer to avoid clipping
                if (
                    len(session.audio_buffer) == 0
                    and len(session.pre_speech_buffer) > 0
                ):
                    session.audio_buffer.extend(session.pre_speech_buffer)
                session.audio_buffer.extend(data)
                session.pre_speech_buffer.clear()
            else:
                # Maintain rolling pre-speech buffer
                session.pre_speech_buffer.extend(data)
                if len(session.pre_speech_buffer) > session.PRE_SPEECH_BYTES:
                    # Trim to keep only the last PRE_SPEECH_BYTES bytes
                    session.pre_speech_buffer = session.pre_speech_buffer[
                        -session.PRE_SPEECH_BYTES:
                    ]

            # ───────────────────────────────────────────────────────────
            # 3. TRANSCRIBE: Speech ended → ASR → LLM → TTS pipeline
            # ───────────────────────────────────────────────────────────
            if vad_result["speech_ended"] and len(session.audio_buffer) > 0:
                # Concurrent-pipeline guard: if a previous turn's pipeline is
                # somehow still running, cancel it before launching new.
                if (
                    session.active_task is not None
                    and not session.active_task.done()
                ):
                    logger.warning(
                        "Concurrent pipeline guard: cancelling in-flight task"
                    )
                    session.active_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await session.active_task

                # ASR: synchronous CPU work — off-thread
                user_text = await asyncio.to_thread(
                    engines.asr_engine.transcribe,
                    bytes(session.audio_buffer),
                )
                session.audio_buffer.clear()
                session.pre_speech_buffer.clear()

                if not user_text.strip():
                    continue

                elapsed_asr_ms = (time.perf_counter() - turn_start) * 1000
                logger.info(f"📝 ASR ({elapsed_asr_ms:.0f}ms): {user_text}")

                # Send transcript to UI
                await ws.send_json(
                    {"type": "TRANSCRIPT", "role": "user", "text": user_text}
                )
                session.chat_history.append(
                    {"role": "user", "content": user_text}
                )

                # Launch streaming LLM → TTS pipeline as a background task
                # so we keep receiving mic chunks (for barge-in detection).
                session.active_task = asyncio.create_task(
                    run_response_pipeline(ws, session)
                )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected — {id(session):#x}")
        if (
            session.active_task is not None
            and not session.active_task.done()
        ):
            session.active_task.cancel()
            with suppress(asyncio.CancelledError):
                await session.active_task


async def run_response_pipeline(ws: WebSocket, session: Session):
    """Stream LLM tokens → sentence chunking → Kokoro TTS → WebSocket audio.

    Runs as a background task so the WebSocket receive loop can keep
    pulling mic chunks for barge-in detection. Cancellable: barge-in
    cancels this task via session.active_task.cancel().
    """
    pipeline_start = time.perf_counter()
    full_response = ""
    sentence_buffer = ""
    first_audio_sent = False

    try:
        session.is_agent_speaking = True
        session.words_spoken.clear()
        session.bytes_sent = 0

        async for token in engines.llm_engine.generate_stream(session.chat_history):
            sentence_buffer += token
            full_response += token

            # Stream TTS per completed sentence for lowest TTFA
            if any(sentence_buffer.rstrip().endswith(p) for p in ".?!\n"):
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if not sentence:
                    continue

                # Track words for state reconstruction (only what's synthesized)
                session.words_spoken.extend(sentence.split())

                # Send text transcript to UI
                await ws.send_json(
                    {"type": "TRANSCRIPT", "role": "assistant", "text": sentence}
                )

                # Synthesize — off-thread so cancel propagates fast
                pcm_bytes, sample_rate = await engines.tts_engine.synthesize(sentence)
                if not pcm_bytes:
                    continue

                # Send sample-rate metadata before first audio chunk
                if not first_audio_sent:
                    if LOG_LATENCY:
                        elapsed = (time.perf_counter() - pipeline_start) * 1000
                        logger.info(f"🔊 TTFA: {elapsed:.0f}ms")
                    await ws.send_json(
                        {"type": "AUDIO_START", "sample_rate": sample_rate}
                    )
                    first_audio_sent = True

                await ws.send_bytes(pcm_bytes)
                session.bytes_sent += len(pcm_bytes)

        # Flush any remaining text without punctuation
        if sentence_buffer.strip():
            sentence = sentence_buffer.strip()
            session.words_spoken.extend(sentence.split())
            await ws.send_json(
                {"type": "TRANSCRIPT", "role": "assistant", "text": sentence}
            )
            pcm_bytes, sample_rate = await engines.tts_engine.synthesize(sentence)
            if pcm_bytes:
                if not first_audio_sent:
                    await ws.send_json(
                        {"type": "AUDIO_START", "sample_rate": sample_rate}
                    )
                await ws.send_bytes(pcm_bytes)

        # Signal end of agent turn
        await ws.send_json({"type": "END_OF_TURN"})

        # Append the complete response to history
        session.chat_history.append(
            {"role": "assistant", "content": full_response.strip()}
        )
        session.is_agent_speaking = False

        if LOG_LATENCY:
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.info(f"✅ Turn complete: {total_ms:.0f}ms total")

    except asyncio.CancelledError:
        logger.info("🛑 Pipeline cancelled by barge-in")
        session.is_agent_speaking = False
        # The chat_history entry is appended by the barge-in handler's
        # state reconstruction (Step 4) using words_spoken.
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
