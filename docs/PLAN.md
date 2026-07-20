# Realtime Voice Tutor — Sub-500ms Voice Conversation Practice Agent

**Repo:** `realtime-voice-tutor`

**Goal:** Build a fully-local, sub-500ms real-time voice agent for **casual English conversation practice**, featuring **Streaming ASR**, **Semantic VAD with Echo Cancellation**, **Agentic LLM Tool Loops (MCP)**, **Local Kokoro TTS**, and **4-Step Full-Duplex Barge-in Interruption with State Reconstruction**.

> 💡 **Everything runs locally** — no cloud dependencies. Silero VAD, Faster-Whisper ASR, Ollama LLM, and Kokoro TTS all run on your machine. To swap in cloud providers (Groq, NVIDIA NIM), just change endpoint URLs in `config.py`.

---

## 🏗️ Architectural Overview

```mermaid
sequenceDiagram
    autonumber
    actor User as User (English Learner)
    participant UI as Browser WebAudio UI
    participant Server as FastAPI Orchestrator
    participant VAD as Silero VAD (Echo-Aware)
    participant ASR as Faster-Whisper ASR
    participant LLM as Ollama llama3.2
    participant MCP as MCP Tool Server
    participant Data as Markdown Scenarios
    participant TTS as Kokoro TTS (Local)

    User->>UI: Speaks audio (16kHz PCM)
    UI->>Server: WebSocket Binary Frames
    Server->>VAD: Speech Probability Check

    alt Active TTS & User Interrupts (Barge-in)
        VAD-->>Server: Speech Prob > 0.92 (raised threshold)
        par 4-Step Interruption
            Server->>UI: FLUSH Signal
            Server->>LLM: Cancel Active Task
            Server->>TTS: Stop Synthesis
            Server->>Server: Reconstruct State (played bytes → chat history)
        end
        UI--xUser: Audio Halts Instantly
    end

    VAD-->>Server: Speech End Detected
    Server->>ASR: Transcribe Audio Buffer
    ASR-->>Server: User Text
    Server->>LLM: Stream Prompt + History + Tools

    opt Tool Call (e.g. get_scenario)
        LLM->>MCP: Execute Tool
        MCP->>Data: Read Markdown + Frontmatter
        Server->>TTS: Filler Audio ("Let me find a good scenario...")
        Data-->>MCP: Filtered Content
        MCP-->>LLM: Tool Result
    end

    LLM-->>Server: Stream Text Tokens
    Server->>TTS: Synthesize Sentence Chunks
    TTS-->>UI: Raw PCM Audio (24kHz)
    UI-->>User: Plays Response (TTFA < 500ms)
```

### ⏱️ Latency Budget (< 450ms TTFA)

| Stage | Component | Target |
| :--- | :--- | :--- |
| **Mic & VAD** | WebAudio + Silero VAD + Browser AEC | **30 ms** |
| **ASR** | Faster-Whisper `tiny.en` (CTranslate2 INT8) | **90 ms** |
| **LLM TTFT** | Ollama `llama3.2:3b` | **180 ms** |
| **TTS First Chunk** | Kokoro ONNX (local, no network) | **60 ms** |
| **Network & Buffer** | Local WebSocket + PCM direct | **30 ms** |
| **Total TTFA** | End-to-end voice loop | **~390 ms** |

---

## 📁 Project Structure

```
realtime-voice-tutor/
├── pyproject.toml              # uv project config & dependencies
├── .env.example                # Environment variable template
├── config.py                   # Centralized config with env overrides
├── server.py                   # FastAPI WebSocket Orchestrator & Barge-in
├── vad_engine.py               # Silero VAD with echo-aware thresholds
├── asr_engine.py               # Faster-Whisper ASR worker
├── llm_engine.py               # Ollama streaming + tool-calling loop
├── tts_engine.py               # Kokoro ONNX local TTS
├── mcp_tools.py                # MCP tool definitions (scenario, phrases, vocab)
├── data_loader.py              # Markdown + frontmatter loader & filter
├── data/
│   ├── scenarios/
│   │   ├── restaurant.md       # Ordering, recommendations, complaints
│   │   ├── airport.md          # Check-in, directions, delays
│   │   ├── workplace.md        # Small talk, meetings, email follow-ups
│   │   ├── shopping.md         # Returns, bargaining, asking for help
│   │   ├── meeting-people.md   # Introductions, hobbies, plans
│   │   └── phone-calls.md      # Appointments, customer service
│   └── vocabulary/
│       ├── a2-beginner.md
│       ├── b1-intermediate.md
│       └── b2-upper.md
├── static/
│   └── index.html              # WebAudio UI with visualizer & barge-in
└── docs/
    └── PLAN.md
```

---

## 🛠️ Phase 1: Foundation & Environment Setup

### 1.1 Dependencies

```bash
cd realtime-voice-tutor
uv init
uv venv .venv --python 3.12
source .venv/bin/activate

uv add fastapi uvicorn[standard] websockets numpy
uv add torch torchaudio                    # Silero VAD runtime
uv add faster-whisper                       # CTranslate2 ASR
uv add kokoro-onnx                          # Local TTS engine
uv add httpx                               # Async HTTP for Ollama API
uv add python-frontmatter                  # Markdown frontmatter parser
uv add python-dotenv                       # .env loading
```

> ⚠️ **Note:** `silero-vad` is NOT a pip package. It loads via `torch.hub` at runtime. Do not add it to dependencies.

### 1.2 Configuration (`config.py`)

```python
# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# --- LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# --- ASR ---
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny.en")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")

# --- VAD ---
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.6"))
VAD_BARGE_IN_THRESHOLD = float(os.getenv("VAD_BARGE_IN_THRESHOLD", "0.92"))
VAD_SILENCE_TIMEOUT_MS = int(os.getenv("VAD_SILENCE_TIMEOUT_MS", "800"))

# --- TTS ---
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "kokoro-v0_19.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "voices.bin")

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8888"))

# --- System Prompt ---
SYSTEM_PROMPT = """You are a friendly English conversation practice tutor called VoiceTutor.

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
```

### 1.3 Environment Template (`.env.example`)

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# ASR
ASR_MODEL_SIZE=tiny.en

# VAD
VAD_THRESHOLD=0.6
VAD_BARGE_IN_THRESHOLD=0.92
VAD_SILENCE_TIMEOUT_MS=800

# TTS
TTS_VOICE=af_heart
TTS_SPEED=1.0

# Server
HOST=0.0.0.0
PORT=8888
```

---

## 📚 Phase 2: Content Data Layer

### 2.1 Scenario Files with Frontmatter

Each scenario is a Markdown file with YAML frontmatter for filtering.

**`data/scenarios/restaurant.md`**

```markdown
---
title: Restaurant & Café
category: restaurant
tags: [food, ordering, casual, social, tipping]
difficulty: [beginner, intermediate]
scenarios: 3
---

# 🍽️ Restaurant & Café

## Scenario 1: Ordering Coffee
- **Difficulty:** Beginner
- **Setting:** You walk into a busy café during morning rush.

### Role Play
You are the customer. VoiceTutor plays the barista.

### Key Phrases
- "Can I get a..." — Standard casual order
- "I'll have..." — Slightly more polished
- "Make that a large" — Changing your order mid-sentence
- "For here or to go?" — You'll hear this, respond naturally
- "Can I also grab a..." — Adding to your order

### Common Mistakes
- ❌ "I want one coffee" → ✅ "Can I get a coffee?"
- ❌ "Two coffee" → ✅ "Two coffees"
- ❌ "Give me..." → ✅ "Could I get..." (sounds more polite)

### Example Dialogue
> **Barista:** Hey! What can I get started for you?
> **You:** Hi, can I get a large iced latte?
> **Barista:** Sure thing! Any flavor shots?
> **You:** Hmm, could you do vanilla?
> **Barista:** You got it. Anything else?
> **You:** That's it, thanks!

---

## Scenario 2: Asking for Recommendations
- **Difficulty:** Intermediate
- **Setting:** You're at a new restaurant, unfamiliar with the menu.

### Role Play
You are dining with a friend. VoiceTutor plays the server.

### Key Phrases
- "What do you recommend?" — Direct and natural
- "What's good here?" — Very casual
- "I'm torn between the... and the..." — Expressing indecision
- "I'll go with..." — Making your final choice
- "Does that come with...?" — Asking about sides/extras

### Common Mistakes
- ❌ "What is delicious?" → ✅ "What's good here?"
- ❌ "I cannot decide" → ✅ "I'm torn between these two"
- ❌ "Bring me the bill" → ✅ "Can I get the check?" (US) / "Could we get the bill?" (UK)

### Example Dialogue
> **Server:** Welcome in! First time here?
> **You:** Yeah actually! What do you recommend?
> **Server:** The pasta is really popular, and the burger's great too.
> **You:** I'm torn between those two. Which one would you go with?
> **Server:** Honestly, the burger. It comes with these amazing fries.
> **You:** Sold. I'll go with the burger then.
```

### 2.2 Vocabulary Files with Frontmatter

**`data/vocabulary/b1-intermediate.md`**

```markdown
---
title: B1 Intermediate Vocabulary
level: B1
tags: [small-talk, opinions, social, work, daily-life]
word_count: 45
---

# B1 Intermediate Vocabulary

## Small Talk & Social
| Word/Phrase | Meaning | Example |
|-------------|---------|---------|
| commute | travel to work regularly | "My commute is about 30 minutes by train" |
| catch up | talk after not seeing someone | "We should catch up over coffee sometime" |
| hang out | spend casual time together | "Want to hang out this weekend?" |
| get along | have a good relationship | "I get along really well with my coworkers" |
| run into | meet someone unexpectedly | "I ran into my old teacher at the store" |

## Opinions & Reactions
| Word/Phrase | Meaning | Example |
|-------------|---------|---------|
| to be honest | being frank (often abbreviated "tbh") | "To be honest, I wasn't a big fan of that movie" |
| I'm into... | I like/enjoy something | "I'm really into hiking lately" |
| not my thing | I don't enjoy it | "Horror movies aren't really my thing" |
| no big deal | it's not important | "Don't worry about it, no big deal" |
| fair enough | I accept that reasoning | "Fair enough, let's try your way" |
```

### 2.3 Data Loader (`data_loader.py`)

```python
# data_loader.py
import frontmatter
from pathlib import Path
from typing import Optional


class DataLoader:
    """Loads and filters Markdown content files with YAML frontmatter."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scenarios: dict[str, dict] = {}
        self.vocabulary: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all markdown files into memory at startup."""
        for f in (self.data_dir / "scenarios").glob("*.md"):
            post = frontmatter.load(f)
            self.scenarios[f.stem] = {
                "meta": post.metadata,
                "content": post.content,
            }

        for f in (self.data_dir / "vocabulary").glob("*.md"):
            post = frontmatter.load(f)
            self.vocabulary[f.stem] = {
                "meta": post.metadata,
                "content": post.content,
            }

    def find_scenarios(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[dict]:
        """Filter scenarios by category, tag, or difficulty."""
        results = []
        for name, data in self.scenarios.items():
            meta = data["meta"]
            if category and meta.get("category") != category:
                continue
            if tag and tag not in meta.get("tags", []):
                continue
            if difficulty and difficulty not in meta.get("difficulty", []):
                continue
            results.append({"name": name, **data})
        return results

    def find_vocabulary(
        self,
        level: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[dict]:
        """Filter vocabulary by CEFR level or tag."""
        results = []
        for name, data in self.vocabulary.items():
            meta = data["meta"]
            if level and meta.get("level") != level:
                continue
            if tag and tag not in meta.get("tags", []):
                continue
            results.append({"name": name, **data})
        return results

    def get_all_categories(self) -> list[str]:
        """Return all available scenario categories."""
        return list({d["meta"].get("category", "") for d in self.scenarios.values()})

    def get_all_tags(self) -> list[str]:
        """Return all unique tags across scenarios."""
        tags = set()
        for d in self.scenarios.values():
            tags.update(d["meta"].get("tags", []))
        return sorted(tags)
```

---

## 🎙️ Phase 3: Semantic VAD Engine (`vad_engine.py`)

Echo-aware VAD that uses higher thresholds during agent speech to prevent false barge-in from speaker feedback.

```python
# vad_engine.py
import torch
import numpy as np
from config import VAD_THRESHOLD, VAD_BARGE_IN_THRESHOLD, VAD_SILENCE_TIMEOUT_MS


class VADEngine:
    """Silero VAD with echo-aware dynamic thresholds."""

    CHUNK_SAMPLES = 512  # Silero accepts 256, 512, or 768 @ 16kHz

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        # Silence tracking for end-of-speech detection
        self.silence_frames = 0
        self.silence_timeout_frames = int(
            VAD_SILENCE_TIMEOUT_MS / (self.CHUNK_SAMPLES / sample_rate * 1000)
        )
        self.is_speaking = False

    def reset(self) -> None:
        self.model.reset_states()
        self.silence_frames = 0
        self.is_speaking = False

    def process_chunk(
        self, pcm_bytes: bytes, agent_is_speaking: bool = False
    ) -> dict:
        """
        Process a 512-sample 16kHz int16 PCM chunk.

        Returns dict with:
            - speech_prob: float (0.0 - 1.0)
            - is_speech: bool (above active threshold)
            - speech_ended: bool (silence timeout reached after speech)
        """
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)

        speech_prob = self.model(tensor, self.sample_rate).item()

        # Echo-aware threshold: raised during agent TTS to reject echo bleed
        threshold = VAD_BARGE_IN_THRESHOLD if agent_is_speaking else VAD_THRESHOLD

        is_speech = speech_prob > threshold

        # Track speech start/end with silence timeout
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
```


---

## ⚡ Phase 4: Streaming ASR Engine (`asr_engine.py`)

```python
# asr_engine.py
import numpy as np
from faster_whisper import WhisperModel
from config import ASR_MODEL_SIZE, ASR_DEVICE


class ASREngine:
    """Faster-Whisper ASR with CTranslate2 INT8 quantization."""

    def __init__(self):
        self.model = WhisperModel(
            ASR_MODEL_SIZE,
            device=ASR_DEVICE,
            compute_type="int8",
        )

    def transcribe(self, pcm_buffer: bytearray) -> str:
        """Transcribe accumulated PCM int16 audio buffer to text."""
        if len(pcm_buffer) < 1600:  # < 100ms, too short
            return ""

        audio = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,  # Built-in VAD to skip silence segments
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text
```

---

## 🤖 Phase 5: Agentic LLM + MCP Tool Loop

### 5.1 MCP Tool Definitions (`mcp_tools.py`)

```python
# mcp_tools.py
import random
from typing import Optional
from data_loader import DataLoader

# Global data store — loaded once at startup
data = DataLoader("data")


def get_scenario(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> str:
    """
    Find a conversation practice scenario.
    Returns scenario content with role-play setup, key phrases, and examples.
    """
    results = data.find_scenarios(category=category, difficulty=difficulty)
    if not results:
        # Fallback: return any random scenario
        results = data.find_scenarios()

    if not results:
        return "No scenarios available. Let's just have a free conversation!"

    chosen = random.choice(results)
    return f"**{chosen['meta']['title']}**\n\n{chosen['content']}"


def lookup_phrases(category: str) -> str:
    """
    Look up useful English phrases for a specific conversation category.
    Returns key phrases with usage notes and common mistakes.
    """
    results = data.find_scenarios(category=category)
    if not results:
        results = data.find_scenarios(tag=category)

    if not results:
        return f"No phrases found for '{category}'. Try: {', '.join(data.get_all_categories())}"

    # Extract just the Key Phrases and Common Mistakes sections
    content = results[0]["content"]
    return content


def check_vocabulary(word: str) -> str:
    """
    Look up an English word or phrase — definition, example usage, and CEFR level.
    """
    for vocab in data.vocabulary.values():
        if word.lower() in vocab["content"].lower():
            level = vocab["meta"].get("level", "unknown")
            return f"[Level: {level}]\n\n{vocab['content']}"

    return f"'{word}' not found in vocabulary database. Try explaining it in context."


def suggest_topic() -> str:
    """Suggest a random conversation topic to practice."""
    categories = data.get_all_categories()
    tags = data.get_all_tags()
    category = random.choice(categories) if categories else "general"
    return (
        f"How about practicing **{category}** conversations?\n"
        f"Available topics: {', '.join(categories)}\n"
        f"Available tags: {', '.join(tags[:10])}"
    )


# --- Tool registry for Ollama tool-calling API ---
TOOL_REGISTRY = {
    "get_scenario": get_scenario,
    "lookup_phrases": lookup_phrases,
    "check_vocabulary": check_vocabulary,
    "suggest_topic": suggest_topic,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_scenario",
            "description": "Find a conversation practice scenario by category or difficulty. Use when the user wants to practice a specific situation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Scenario category: restaurant, airport, workplace, shopping, meeting-people, phone-calls",
                    },
                    "difficulty": {
                        "type": "string",
                        "description": "Difficulty level: beginner, intermediate, advanced",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_phrases",
            "description": "Look up useful English phrases for a conversation category. Use when the user is stuck or asks how to say something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The conversation category to find phrases for",
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_vocabulary",
            "description": "Look up an English word or phrase — definition, examples, and CEFR level. Use when the user asks about a word or uses one incorrectly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The English word or phrase to look up",
                    },
                },
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_topic",
            "description": "Suggest a random conversation topic to practice. Use when the user doesn't know what to practice.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]
```

### 5.2 Streaming LLM Engine with Tool Loop (`llm_engine.py`)

```python
# llm_engine.py
import httpx
import json
from typing import AsyncIterator
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPT
from mcp_tools import TOOL_SCHEMAS, TOOL_REGISTRY


class LLMEngine:
    """Async streaming LLM with Ollama tool-calling loop."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """
        Stream LLM response tokens. Handles tool calls automatically:
        if the model requests a tool, executes it and continues generation.

        Yields text tokens as they arrive.
        """
        # Ensure system prompt is first
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep last 10 turns for context window management
        full_messages.extend(messages[-10:])

        max_tool_rounds = 3  # Prevent infinite tool loops

        for _ in range(max_tool_rounds):
            payload = {
                "model": OLLAMA_MODEL,
                "messages": full_messages,
                "tools": TOOL_SCHEMAS,
                "stream": True,
            }

            accumulated_text = ""
            tool_calls = []

            async with self.client.stream(
                "POST", self.chat_url, json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    msg = chunk.get("message", {})

                    # Yield streamed text tokens
                    token = msg.get("content", "")
                    if token:
                        accumulated_text += token
                        yield token

                    # Collect tool calls
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])

            # If no tool calls, we're done
            if not tool_calls:
                return

            # Execute tool calls and add results to messages
            full_messages.append({
                "role": "assistant",
                "content": accumulated_text,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"].get("arguments", {})
                handler = TOOL_REGISTRY.get(func_name)

                if handler:
                    result = handler(**func_args)
                else:
                    result = f"Unknown tool: {func_name}"

                full_messages.append({
                    "role": "tool",
                    "content": str(result),
                })

            # Clear for next round — LLM will generate response using tool results
            tool_calls = []

    async def close(self):
        await self.client.aclose()
```


---

## 🔊 Phase 6: Local Kokoro TTS Engine (`tts_engine.py`)

Fully local TTS — no cloud, no network latency, deterministic performance.

```python
# tts_engine.py
import numpy as np
from kokoro_onnx import Kokoro
from config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, TTS_VOICE, TTS_SPEED


class TTSEngine:
    """Kokoro ONNX local text-to-speech engine."""

    def __init__(self):
        self.kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
        self.voice = TTS_VOICE
        self.speed = TTS_SPEED
        self.sample_rate = 24000  # Kokoro outputs 24kHz audio

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """
        Synthesize text to raw PCM int16 bytes.

        Returns:
            (pcm_bytes, sample_rate) — ready to send over WebSocket
        """
        if not text.strip():
            return b"", self.sample_rate

        audio_float32, sr = self.kokoro.create(
            text, voice=self.voice, speed=self.speed
        )

        # Convert float32 [-1, 1] to int16 PCM bytes
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        return audio_int16.tobytes(), sr

    def synthesize_filler(self) -> tuple[bytes, int]:
        """Generate a short filler audio for tool execution delays."""
        fillers = [
            "Let me check on that.",
            "One moment.",
            "Looking that up for you.",
        ]
        import random
        return self.synthesize(random.choice(fillers))
```

**Why Kokoro over edge-tts:**
- ✅ ~60ms synthesis vs ~300ms (no network round-trip)
- ✅ No Microsoft cloud dependency
- ✅ Raw PCM output (no MP3 decode overhead)
- ✅ Deterministic latency (no cloud variability)
- ✅ Works offline

---

## 🧠 Phase 7: WebSocket Orchestrator & 4-Step Barge-in (`server.py`)

This is the **heart of the POC** — tying all engines together with the barge-in interruption flow.

```python
# server.py
import asyncio
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from vad_engine import VADEngine
from asr_engine import ASREngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from config import HOST, PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicetutor")

app = FastAPI(title="Realtime Voice Tutor")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


class Session:
    """Per-connection state — each browser tab gets its own session."""

    def __init__(self):
        self.vad = VADEngine()
        self.asr = ASREngine()
        self.llm = LLMEngine()
        self.tts = TTSEngine()

        self.chat_history: list[dict] = []
        self.active_task: asyncio.Task | None = None
        self.is_agent_speaking: bool = False

        # Barge-in state reconstruction
        self.words_spoken: list[str] = []
        self.bytes_sent: int = 0

        # Audio accumulation
        self.audio_buffer = bytearray()

        # Pre-speech padding: keep last 200ms of audio before VAD triggers
        # to avoid clipping the start of the user's utterance
        self.pre_speech_buffer = bytearray()
        self.PRE_SPEECH_BYTES = 16000 * 2 * 200 // 1000  # 200ms of 16kHz int16


@app.websocket("/ws/voice")
async def voice_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    logger.info("Client connected — new session created")

    try:
        while True:
            data = await ws.receive_bytes()
            turn_start = time.perf_counter()

            vad_result = session.vad.process_chunk(
                data, agent_is_speaking=session.is_agent_speaking
            )

            # ──────────────────────────────────────────────
            # 1. BARGE-IN: User interrupts during agent TTS
            # ──────────────────────────────────────────────
            if session.is_agent_speaking and vad_result["is_speech"]:
                logger.info("⚡ BARGE-IN DETECTED — 4-Step Interruption")

                # Step 1: Flush client audio buffer immediately
                await ws.send_json({"type": "FLUSH"})

                # Step 2: Cancel active LLM + TTS pipeline
                if session.active_task and not session.active_task.done():
                    session.active_task.cancel()

                # Step 3: Halt agent speaking state
                session.is_agent_speaking = False

                # Step 4: State Reconstruction
                # Trim chat history to reflect only what was actually spoken
                spoken_text = " ".join(session.words_spoken)
                if (
                    session.chat_history
                    and session.chat_history[-1]["role"] == "assistant"
                ):
                    session.chat_history[-1]["content"] = (
                        spoken_text + " [interrupted]"
                    )
                logger.info(f"State reconstructed: '{spoken_text[:80]}...'")

                # Reset for new user turn
                session.audio_buffer.clear()
                session.words_spoken.clear()
                session.bytes_sent = 0
                session.vad.reset()
                continue

            # ──────────────────────────────────────────────
            # 2. ACCUMULATE: Buffer user audio during speech
            # ──────────────────────────────────────────────
            if vad_result["is_speech"]:
                # On speech start, prepend pre-speech buffer to avoid clipping
                if len(session.audio_buffer) == 0 and len(session.pre_speech_buffer) > 0:
                    session.audio_buffer.extend(session.pre_speech_buffer)
                session.audio_buffer.extend(data)
            else:
                # Maintain rolling pre-speech buffer
                session.pre_speech_buffer.extend(data)
                if len(session.pre_speech_buffer) > session.PRE_SPEECH_BYTES:
                    session.pre_speech_buffer = session.pre_speech_buffer[
                        -session.PRE_SPEECH_BYTES:
                    ]

            # ──────────────────────────────────────────────
            # 3. TRANSCRIBE: Speech ended → ASR → LLM → TTS
            # ──────────────────────────────────────────────
            if vad_result["speech_ended"] and len(session.audio_buffer) > 0:
                user_text = session.asr.transcribe(session.audio_buffer)
                session.audio_buffer.clear()
                session.pre_speech_buffer.clear()

                if not user_text.strip():
                    continue

                elapsed_asr = (time.perf_counter() - turn_start) * 1000
                logger.info(
                    f"📝 ASR ({elapsed_asr:.0f}ms): {user_text}"
                )

                # Send transcript to UI
                await ws.send_json(
                    {"type": "TRANSCRIPT", "role": "user", "text": user_text}
                )
                session.chat_history.append(
                    {"role": "user", "content": user_text}
                )

                # Launch streaming LLM → TTS pipeline
                session.active_task = asyncio.create_task(
                    run_response_pipeline(ws, session)
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        await session.llm.close()


async def run_response_pipeline(ws: WebSocket, session: Session):
    """Stream LLM tokens → sentence chunking → Kokoro TTS → WebSocket audio."""
    try:
        session.is_agent_speaking = True
        session.words_spoken.clear()
        session.bytes_sent = 0

        sentence_buffer = ""
        full_response = ""
        turn_start = time.perf_counter()
        first_audio_sent = False

        async for token in session.llm.generate_stream(session.chat_history):
            sentence_buffer += token
            full_response += token

            # Stream TTS per completed sentence for lowest TTFA
            if any(sentence_buffer.rstrip().endswith(p) for p in ".?!\n"):
                sentence = sentence_buffer.strip()
                if not sentence:
                    sentence_buffer = ""
                    continue

                # Track words for state reconstruction
                session.words_spoken.extend(sentence.split())

                # Send text transcript to UI
                await ws.send_json(
                    {"type": "TRANSCRIPT", "role": "assistant", "text": sentence}
                )

                # Synthesize and send audio
                pcm_bytes, sample_rate = session.tts.synthesize(sentence)
                if pcm_bytes:
                    # Send sample rate metadata before first audio chunk
                    if not first_audio_sent:
                        elapsed = (time.perf_counter() - turn_start) * 1000
                        logger.info(f"🔊 TTFA: {elapsed:.0f}ms")
                        await ws.send_json(
                            {"type": "AUDIO_START", "sample_rate": sample_rate}
                        )
                        first_audio_sent = True

                    await ws.send_bytes(pcm_bytes)
                    session.bytes_sent += len(pcm_bytes)

                sentence_buffer = ""

        # Handle any remaining text that didn't end with punctuation
        if sentence_buffer.strip():
            sentence = sentence_buffer.strip()
            session.words_spoken.extend(sentence.split())
            await ws.send_json(
                {"type": "TRANSCRIPT", "role": "assistant", "text": sentence}
            )
            pcm_bytes, sample_rate = session.tts.synthesize(sentence)
            if pcm_bytes:
                if not first_audio_sent:
                    await ws.send_json(
                        {"type": "AUDIO_START", "sample_rate": sample_rate}
                    )
                await ws.send_bytes(pcm_bytes)

        # Signal end of agent turn
        await ws.send_json({"type": "END_OF_TURN"})

        # Append complete response to history
        session.chat_history.append(
            {"role": "assistant", "content": full_response.strip()}
        )
        session.is_agent_speaking = False

        total_ms = (time.perf_counter() - turn_start) * 1000
        logger.info(f"✅ Turn complete: {total_ms:.0f}ms total")

    except asyncio.CancelledError:
        logger.info("🛑 Pipeline cancelled by barge-in")
        session.is_agent_speaking = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
```


---

## 🎨 Phase 8: Browser UI (`static/index.html`)

Dark-mode dashboard with proper sequential audio playback, real-time visualizer, and instant barge-in buffer flushing.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Realtime Voice Tutor — English Practice</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: #0a0e1a;
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 24px;
        }

        .container {
            width: 100%;
            max-width: 720px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* --- Header --- */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 24px;
            background: linear-gradient(135deg, #1a1f35, #1e2744);
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.15);
        }

        .header h1 {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #818cf8, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(71, 85, 105, 0.4);
            color: #94a3b8;
            transition: all 0.3s ease;
        }

        .status-badge.connected {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.1);
        }

        .status-badge.speaking {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        /* --- Chat --- */
        .chat-box {
            flex: 1;
            min-height: 400px;
            max-height: 60vh;
            overflow-y: auto;
            background: #0f1322;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .msg {
            max-width: 80%;
            padding: 10px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.6;
            animation: fadeIn 0.2s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg.user {
            align-self: flex-end;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.2);
            color: #7dd3fc;
        }

        .msg.assistant {
            align-self: flex-start;
            background: rgba(74, 222, 128, 0.08);
            border: 1px solid rgba(74, 222, 128, 0.15);
            color: #86efac;
        }

        .msg.system {
            align-self: center;
            background: transparent;
            color: #475569;
            font-size: 12px;
            font-style: italic;
        }

        /* --- Controls --- */
        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .btn {
            flex: 1;
            padding: 14px 24px;
            border: none;
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-primary.active {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        }

        /* --- Latency Display --- */
        .latency {
            text-align: center;
            font-size: 11px;
            color: #475569;
        }

        .latency span {
            color: #6366f1;
            font-weight: 600;
        }

        /* Scrollbar */
        .chat-box::-webkit-scrollbar { width: 6px; }
        .chat-box::-webkit-scrollbar-track { background: transparent; }
        .chat-box::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗣️ Voice Tutor</h1>
            <div class="status-badge" id="status">Disconnected</div>
        </div>

        <div class="chat-box" id="chat">
            <div class="msg system">Click "Start Talking" and say something to begin practicing!</div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" id="startBtn" onclick="toggleVoice()">
                Start Talking
            </button>
        </div>
        <div class="latency" id="latency"></div>
    </div>

    <script>
        let ws = null;
        let audioCtx = null;
        let micStream = null;
        let processor = null;

        // --- Sequential Audio Queue ---
        // Unlike v1 which created overlapping Audio() elements,
        // this queues PCM buffers and plays them in sequence via AudioContext.
        let audioQueue = [];
        let isPlayingAudio = false;
        let currentSource = null;
        let ttsSampleRate = 24000;

        async function toggleVoice() {
            const btn = document.getElementById('startBtn');

            if (ws) {
                ws.close();
                stopMic();
                btn.textContent = 'Start Talking';
                btn.classList.remove('active');
                return;
            }

            ws = new WebSocket(`ws://${location.host}/ws/voice`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                setStatus('connected', 'Connected & Listening');
                btn.textContent = 'Stop';
                btn.classList.add('active');
                startMic();
            };

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);

                    switch (msg.type) {
                        case 'FLUSH':
                            // Barge-in: immediately stop all audio
                            flushAudio();
                            break;

                        case 'TRANSCRIPT':
                            appendMessage(msg.role, msg.text);
                            if (msg.role === 'assistant') {
                                setStatus('speaking', 'Agent Speaking...');
                            }
                            break;

                        case 'AUDIO_START':
                            ttsSampleRate = msg.sample_rate || 24000;
                            break;

                        case 'END_OF_TURN':
                            setStatus('connected', 'Connected & Listening');
                            break;
                    }
                } else {
                    // Binary = PCM audio chunk from Kokoro TTS
                    queueAudio(event.data);
                }
            };

            ws.onclose = () => {
                setStatus('disconnected', 'Disconnected');
                btn.textContent = 'Start Talking';
                btn.classList.remove('active');
                ws = null;
                flushAudio();
            };

            ws.onerror = () => {
                setStatus('disconnected', 'Connection Error');
            };
        }

        // --- Microphone Capture ---
        async function startMic() {
            audioCtx = new AudioContext({ sampleRate: 16000 });

            // Browser AEC: echoCancellation removes speaker feedback from mic input
            // This prevents false barge-in triggers from agent's own TTS audio
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000,
                }
            });

            const source = audioCtx.createMediaStreamSource(micStream);

            // ScriptProcessor sends 512-sample chunks matching Silero VAD input size
            // Note: ScriptProcessor is deprecated; AudioWorkletNode is preferred for production
            processor = audioCtx.createScriptProcessor(512, 1, 1);
            processor.onaudioprocess = (e) => {
                if (!ws || ws.readyState !== WebSocket.OPEN) return;
                const float32 = e.inputBuffer.getChannelData(0);
                const int16 = new Int16Array(float32.length);
                for (let i = 0; i < float32.length; i++) {
                    int16[i] = Math.max(-1, Math.min(1, float32[i])) * 0x7FFF;
                }
                ws.send(int16.buffer);
            };

            source.connect(processor);
            processor.connect(audioCtx.destination);
        }

        function stopMic() {
            if (processor) { processor.disconnect(); processor = null; }
            if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
            if (audioCtx) { audioCtx.close(); audioCtx = null; }
        }

        // --- Sequential Audio Playback ---
        // Queues PCM int16 buffers and plays them back-to-back through AudioContext
        // without overlap or gaps.

        let playbackCtx = null;
        let nextPlayTime = 0;

        function queueAudio(arrayBuffer) {
            if (!playbackCtx) {
                playbackCtx = new AudioContext({ sampleRate: ttsSampleRate });
                nextPlayTime = playbackCtx.currentTime;
            }

            const int16 = new Int16Array(arrayBuffer);
            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) {
                float32[i] = int16[i] / 32768.0;
            }

            const buffer = playbackCtx.createBuffer(1, float32.length, ttsSampleRate);
            buffer.getChannelData(0).set(float32);

            const source = playbackCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(playbackCtx.destination);

            // Schedule seamlessly after previous chunk
            const startTime = Math.max(nextPlayTime, playbackCtx.currentTime);
            source.start(startTime);
            nextPlayTime = startTime + buffer.duration;

            // Track for flush
            audioQueue.push(source);
            source.onended = () => {
                const idx = audioQueue.indexOf(source);
                if (idx > -1) audioQueue.splice(idx, 1);
            };
        }

        function flushAudio() {
            console.log('⚡ FLUSH — stopping all audio');
            // Stop all scheduled and playing audio sources
            audioQueue.forEach(source => {
                try { source.stop(); } catch(e) {}
            });
            audioQueue = [];
            nextPlayTime = 0;

            // Reset playback context to clear any scheduled buffers
            if (playbackCtx) {
                playbackCtx.close();
                playbackCtx = null;
            }
        }

        // --- UI Helpers ---
        function setStatus(state, text) {
            const el = document.getElementById('status');
            el.textContent = text;
            el.className = 'status-badge ' + state;
        }

        function appendMessage(role, text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
```


---

## 🚀 Phase 9: Run & Test

### 1. Download Kokoro Model Files

```bash
# Download ONNX model and voice pack (one-time setup)
pip install huggingface-hub
huggingface-cli download hexgrad/Kokoro-82M --include 'kokoro-v0_19.onnx' 'voices.bin' --local-dir .
```

### 2. Start Ollama

```bash
ollama pull llama3.2:3b
ollama serve
```

### 3. Launch the Server

```bash
cp .env.example .env
python server.py
```

### 4. Test in Browser

1. Open `http://localhost:8888` in Chrome.
2. Click **"Start Talking"** and say: *"Hi! I want to practice ordering food at a restaurant."*
3. The agent should call `get_scenario(category="restaurant")` and start a role-play.
4. **Test barge-in:** While the agent is speaking, interrupt with *"Wait, how do I say that more naturally?"*
5. Observe:
   - Audio stops instantly (< 50ms)
   - Server logs `⚡ BARGE-IN DETECTED`
   - Agent responds to your interruption with context preserved

### 5. Test Scenarios

| Test | What to Say | Expected Behavior |
|------|-------------|-------------------|
| Scenario start | "Let's practice something" | Agent calls `suggest_topic()` |
| Specific topic | "Can we do airport conversations?" | Agent calls `get_scenario(category="airport")` |
| Vocabulary help | "What does 'commute' mean?" | Agent calls `check_vocabulary(word="commute")` |
| Barge-in | Interrupt mid-sentence | Audio stops, state reconstructed |
| Short utterance | "Yes" or "OK" | Transcribed despite < 1 sec duration |
| Natural flow | Free conversation | Agent corrects grammar gently |

---

## 🎯 Benchmark Verification Checklist

- [ ] **VAD Detection:** < 30ms (Silero ONNX, 512-sample chunks)
- [ ] **Echo Rejection:** No false barge-in when agent speaks through speakers
- [ ] **ASR Transcription:** < 100ms (Faster-Whisper INT8, `tiny.en`)
- [ ] **LLM TTFT:** < 200ms (Ollama `llama3.2:3b`)
- [ ] **Tool Execution:** < 50ms (in-memory markdown lookup)
- [ ] **TTS First Audio:** < 80ms (Kokoro ONNX local)
- [ ] **Barge-in Flush:** < 50ms (WebSocket FLUSH + AudioContext stop)
- [ ] **Total TTFA:** **~390ms** (well under 500ms target)
- [ ] **Short utterance detection:** Works for < 1 sec speech
- [ ] **Pre-speech padding:** First word not clipped

---

## 📌 Known Limitations & Future Improvements

### Current Limitations (acceptable for POC)
- `ScriptProcessor` is deprecated — swap to `AudioWorkletNode` for production
- No WebSocket authentication — POC only, not production-safe
- No conversation persistence — chat history lost on page refresh
- Single language (English) — no multi-language support yet
- No interim ASR results — transcription only after speech ends

### Future Enhancements
- [ ] WebRTC transport for lower latency + built-in echo cancellation
- [ ] Opus codec for mic audio (5x bandwidth reduction)
- [ ] Conversation persistence (save/load sessions)
- [ ] Multi-language support (swap ASR model + TTS voice per locale)
- [ ] Real-time audio waveform visualizer
- [ ] Client-side `bytes_played` reporting for precise state reconstruction
- [ ] Latency dashboard with per-turn timing charts
- [ ] Docker Compose deployment (voice-agent + ollama)
