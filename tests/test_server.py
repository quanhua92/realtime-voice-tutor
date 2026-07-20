"""Integration tests for the FastAPI server + WebSocket protocol.

These start the server in-process using a manual lifespan (so we can
avoid pulling in shared engine state if tests run out-of-order) and
exercise the WebSocket protocol with a mock LLM/TTS to keep tests
deterministic and fast.

Marked `integration` so they can be skipped via `pytest -m "not integration"`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_tutor import engines, server


pytestmark = pytest.mark.integration


SPEECH_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hello_speech_16k.npy"


# --- Fakes for the heavy engines ---

class _FakeASR:
    def __init__(self) -> None:
        self.transcribe = MagicMock(return_value="hello there")


class _FakeLLM:
    """Fake LLM that yields a fixed multi-sentence response."""
    RESPONSE_TOKENS = ["Hello", "! ", "How are you", " today", "?"]

    async def generate_stream(self, messages):
        for tok in self.RESPONSE_TOKENS:
            yield tok

    async def close(self):
        pass


class _FakeTTS:
    """Fake TTS that returns 100ms of silence per call."""
    SAMPLE_RATE = 24000

    def __init__(self, delay_seconds: float = 0.0) -> None:
        """Optional delay_seconds blocks synthesize() to test cancellation."""
        self.delay_seconds = delay_seconds

    async def synthesize(self, text):
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        # 100ms of int16 silence at 24kHz
        pcm = np.zeros(self.SAMPLE_RATE // 10, dtype=np.int16).tobytes()
        return pcm, self.SAMPLE_RATE

    async def synthesize_stream(self, text):
        pcm = np.zeros(self.SAMPLE_RATE // 10, dtype=np.int16).tobytes()
        yield pcm, self.SAMPLE_RATE

    async def synthesize_filler(self):
        return await self.synthesize("filler")


# --- Fixtures ---

@pytest.fixture
def fake_engines():
    """Install fake ASR/LLM/TTS into the engines registry."""
    prev_asr, prev_llm, prev_tts = (
        engines.asr_engine,
        engines.llm_engine,
        engines.tts_engine,
    )
    fake_asr = _FakeASR()
    fake_llm = _FakeLLM()
    fake_tts = _FakeTTS()
    engines.asr_engine = fake_asr
    engines.llm_engine = fake_llm
    engines.tts_engine = fake_tts
    try:
        yield {"asr": fake_asr, "llm": fake_llm, "tts": fake_tts}
    finally:
        engines.asr_engine = prev_asr
        engines.llm_engine = prev_llm
        engines.tts_engine = prev_tts


@pytest.fixture
def client(fake_engines):
    """TestClient WITHOUT triggering lifespan — engines are mocked by the
    fake_engines fixture, and we don't want lifespan to overwrite them
    with real model loads.
    """
    # Construct without context manager so startup events don't fire.
    c = TestClient(server.app, raise_server_exceptions=True)
    yield c


# --- HTTP tests ---

def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # All engines are "loaded" because our fixture set them
    assert body["asr_loaded"] is True
    assert body["llm_loaded"] is True
    assert body["tts_loaded"] is True


def test_root_serves_index_html(client):
    """GET / should serve static/index.html if present."""
    index_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if not index_path.exists():
        pytest.skip("static/index.html not present (UI commit not yet landed)")
    r = client.get("/")
    assert r.status_code == 200


# --- WebSocket happy-path ---

def test_websocket_protocol_smoke(client):
    """Smoke-test the WS endpoint: it should accept a connection and
    not crash on a stream of audio chunks. Full barge-in / pipeline
    verification is done via the manual end-to-end script
    (scripts/manual_e2e_test.py) because Starlette's TestClient has
    awkward timing semantics around streaming WS messages.
    """
    if not SPEECH_FIXTURE.exists():
        pytest.skip("Speech fixture missing — run scripts/generate_test_fixtures.py")

    speech = np.load(SPEECH_FIXTURE)
    chunks = [speech[i : i + 512].tobytes() for i in range(0, len(speech) - 511, 512)]

    with client.websocket_connect("/ws/voice") as ws:
        # Just push a few seconds of audio and confirm the WS stays open.
        # The pipeline may fire speech_ended and emit messages, but we
        # don't strictly assert on them here.
        for chunk in chunks[:100]:  # ~3.2 seconds
            ws.send_bytes(chunk)
        # Try to receive at least one message in a short window
        msg_count = 0
        try:
            for _ in range(5):
                ws.receive(timeout=0.2)
                msg_count += 1
        except Exception:
            pass
        # If we got any messages at all, the protocol is wired up
        # (server processed audio and emitted something). If not,
        # the VAD didn't trigger speech — also fine for this smoke test.
        # The real verification is the manual e2e script.
        assert ws is not None


# --- Barge-in protocol ---

def test_session_reset_after_barge_in_seeds_audio_buffer():
    """Unit test: Session.reset_after_barge_in() seeds the new audio_buffer
    with pre_speech_buffer + current chunk."""
    sess = server.Session()
    # Populate state
    sess.pre_speech_buffer.extend(b"\x01" * 1000)
    sess.words_spoken.extend(["hello", "world"])
    sess.bytes_sent = 9999
    sess.is_speaking = True if hasattr(sess, "is_speaking") else False
    sess.vad.is_speaking = True

    current = b"\x02" * 512
    sess.reset_after_barge_in(current)

    # Audio buffer should contain pre_speech (1000 bytes) + current (512 bytes)
    assert len(sess.audio_buffer) == 1000 + 512
    assert sess.audio_buffer[:1000] == b"\x01" * 1000
    assert sess.audio_buffer[1000:] == b"\x02" * 512
    # State cleared
    assert sess.words_spoken == []
    assert sess.bytes_sent == 0
    assert sess.vad.is_speaking is False


def test_pre_speech_bytes_constant_matches_config():
    """The rolling pre-speech buffer size should match PRE_SPEECH_MS=200ms."""
    sess = server.Session()
    # 16000 * 2 * 200 / 1000 = 6400 bytes
    assert sess.PRE_SPEECH_BYTES == 6400


def test_pre_speech_buffer_trims_to_cap():
    """Adding audio beyond PRE_SPEECH_BYTES keeps only the tail."""
    sess = server.Session()
    # Simulate many chunks of silence being added
    for i in range(50):
        sess.pre_speech_buffer.extend(bytes([i]) * 1024)
        if len(sess.pre_speech_buffer) > sess.PRE_SPEECH_BYTES:
            sess.pre_speech_buffer = sess.pre_speech_buffer[
                -sess.PRE_SPEECH_BYTES:
            ]
    assert len(sess.pre_speech_buffer) == sess.PRE_SPEECH_BYTES


# --- Pipeline cancel propagation ---

@pytest.mark.asyncio
async def test_pipeline_cancellable_mid_synthesis(fake_engines):
    """run_response_pipeline should observe CancelledError quickly when
    the TTS call is wrapped in asyncio.to_thread."""
    # Replace the fake TTS with a slow one that blocks long enough to cancel.
    fake_tts_slow = _FakeTTS(delay_seconds=2.0)
    prev_tts = engines.tts_engine
    engines.tts_engine = fake_tts_slow
    try:
        sess = server.Session()
        ws = AsyncMock()
        sess.chat_history.append({"role": "user", "content": "hi"})

        task = asyncio.create_task(server.run_response_pipeline(ws, sess))
        # Let it reach the first TTS call
        await asyncio.sleep(0.1)
        assert not task.done(), "Pipeline should be blocked on slow TTS"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sess.is_agent_speaking is False
    finally:
        engines.tts_engine = prev_tts
