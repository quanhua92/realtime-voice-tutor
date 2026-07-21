"""Async streaming LLM client with OpenAI-compatible tool-calling loop.

Works with any provider that exposes POST /v1/chat/completions with
streaming + tools: Ollama (default), Groq, NVIDIA NIM, OpenAI itself.
Swap providers by changing OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .config import (
    LLM_HISTORY_TURNS,
    LLM_MAX_TOOL_ROUNDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    TOOLS_ENABLED,
)
from .mcp_tools import TOOL_REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger("voicetutor.llm")


class LLMEngine:
    """Async streaming LLM (OpenAI-compatible) with tool-calling loop."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            timeout=30.0,
        )
        self.model = OPENAI_MODEL

    async def generate_stream(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Stream LLM response tokens; handle tool calls automatically.

        Prepends SYSTEM_PROMPT, trims to the last LLM_HISTORY_TURNS messages,
        and loops up to LLM_MAX_TOOL_ROUNDS times if the model invokes tools.

        Yields:
            Text tokens (str) as they arrive.
        """
        full_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        full_messages.extend(messages[-LLM_HISTORY_TURNS:])

        # If tools are disabled, do a single streaming call with no tools.
        # Small models like qwen3:0.6b hallucinate fake tool-call syntax
        # as text when given the schema — disabling tools entirely avoids
        # that failure mode.
        if not TOOLS_ENABLED:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
            return

        for round_idx in range(LLM_MAX_TOOL_ROUNDS):
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=TOOL_SCHEMAS,
                stream=True,
            )

            accumulated_text = ""
            # OpenAI streams tool_calls in pieces keyed by `index`. We
            # accumulate id/name/arguments per slot.
            tool_calls: dict[int, dict[str, str]] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Stream text tokens to caller as they arrive
                if delta.content:
                    accumulated_text += delta.content
                    yield delta.content

                # Accumulate tool_calls across chunks
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        slot = tool_calls.setdefault(
                            idx, {"id": "", "name": "", "arguments_str": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments_str"] += tc.function.arguments

            # No tool calls this round → response is complete
            if not tool_calls:
                return

            # Append the assistant message that contained the tool_calls.
            # OpenAI requires the assistant message to echo back the tool_calls
            # it produced, with arguments as a JSON string.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_text or None,
                "tool_calls": [
                    {
                        "id": slot["id"],
                        "type": "function",
                        "function": {
                            "name": slot["name"],
                            "arguments": slot["arguments_str"] or "{}",
                        },
                    }
                    for slot in tool_calls.values()
                ],
            }
            full_messages.append(assistant_msg)

            # Dispatch each tool off-thread and append the OpenAI-format result
            for slot in tool_calls.values():
                func_name = slot["name"]
                try:
                    func_args = json.loads(slot["arguments_str"] or "{}")
                    if not isinstance(func_args, dict):
                        func_args = {}
                except json.JSONDecodeError:
                    logger.warning(
                        f"Tool {func_name} returned malformed JSON arguments: "
                        f"{slot['arguments_str']!r}"
                    )
                    func_args = {}

                handler = TOOL_REGISTRY.get(func_name)
                if handler is None:
                    result = f"Unknown tool: {func_name}"
                else:
                    # Tools may do non-trivial work (file reads, etc.) —
                    # run off-thread so the event loop stays responsive.
                    result = await asyncio.to_thread(
                        _safe_call, handler, func_args
                    )

                logger.info(
                    f"🛠️ Tool call: {func_name}({func_args}) → "
                    f"{str(result)[:80]}..."
                )

                # OpenAI tool result format
                full_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": slot["id"],
                        "content": str(result),
                    }
                )

            # Loop continues: LLM will synthesize its next response using
            # the tool results. It may emit text (yielded to caller) or
            # invoke another tool.

        # Safety: if we exhausted max rounds, log it
        logger.warning(
            f"LLM exceeded LLM_MAX_TOOL_ROUNDS={LLM_MAX_TOOL_ROUNDS}; "
            "stopping tool loop"
        )

    async def close(self) -> None:
        await self.client.close()


def _safe_call(handler, args: dict) -> str:
    """Call a tool handler with the given args, catching errors.

    Tools are responsible for their own internal error handling; this
    is a backstop so one buggy tool doesn't kill the whole pipeline.
    """
    try:
        return str(handler(**args))
    except TypeError:
        # Tool doesn't accept these kwargs — try with no args
        try:
            return str(handler())
        except Exception as e:
            return f"Tool error: {e}"
    except Exception as e:
        return f"Tool error: {e}"
