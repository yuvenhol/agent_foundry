"""LangChain streaming -> AG-UI events.

We use LangChain's `astream(..., stream_mode=...)` APIs (preferred) rather than
`astream_events`, per LangChain streaming docs:
https://docs.langchain.com/oss/python/langchain/streaming
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core.events import (
    BaseEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from .ids import new_id


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _extract_chunk_text(chunk: Any) -> str:
    """Best-effort extraction of token delta text from LangChain chunks."""
    if chunk is None:
        return ""

    # LangChain message chunks often have .content (str or list).
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # multimodal chunks are list of dicts/parts; best effort concatenate text-ish parts
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # common pattern: {"type":"text","text":"..."}
                t = p.get("text") if isinstance(p.get("text"), str) else None
                if t:
                    parts.append(t)
        return "".join(parts)

    # Sometimes chunk itself is a string.
    if isinstance(chunk, str):
        return chunk

    return ""


def _get_langgraph_node(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    node = metadata.get("langgraph_node")
    return str(node) if node is not None else None


def _iter_content_blocks(token: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of content_blocks from LangChain streamed tokens."""
    content_blocks = getattr(token, "content_blocks", None)
    if isinstance(content_blocks, list):
        return [b for b in content_blocks if isinstance(b, dict)]

    # Some tokens are AIMessageChunk with .content_blocks property; otherwise fallback to .content.
    content = getattr(token, "content", None)
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


def _text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for b in blocks:
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "".join(parts)


def _tool_call_chunks_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in blocks if b.get("type") == "tool_call_chunk"]


async def langchain_astream_events_to_agui_events(
    *,
    runnable: Any,
    input: dict[str, Any],
    config: dict[str, Any] | None = None,
    thread_id: str | None = None,
    run_id: str | None = None,
) -> AsyncIterator[BaseEvent]:
    """Run a runnable with astream_events and translate into AG-UI events.

    Assumptions:
    - We treat each chat-model generation as a single assistant text message stream.
    - Tool calls are mapped from on_tool_* events when present.
    """
    # Backwards-compat name kept to avoid widespread refactors; implementation uses `astream`.

    thread_id = thread_id or new_id("thread")
    run_id = run_id or new_id("run")

    message_id: str | None = None
    current_node: str | None = None

    active_tool_call_id: str | None = None
    active_tool_call_name: str | None = None
    last_tool_call_id: str | None = None  # for linking tool outputs to the most recent call

    yield RunStartedEvent(thread_id=thread_id, run_id=run_id, timestamp=_now_ms())

    def _step_switch(new_node: str | None) -> list[BaseEvent]:
        nonlocal current_node
        if new_node == current_node:
            return []
        out: list[BaseEvent] = []
        if current_node is not None:
            out.append(StepFinishedEvent(step_name=current_node, timestamp=_now_ms()))
        if new_node is not None:
            out.append(StepStartedEvent(step_name=new_node, timestamp=_now_ms()))
        current_node = new_node
        return out

    try:
        # Prefer "messages" mode for token/tool-call chunks + tool node outputs.
        async for token, metadata in runnable.astream(  # type: ignore[attr-defined]
            input,
            config=config,
            stream_mode="messages",
        ):
            node = _get_langgraph_node(metadata)
            for step_ev in _step_switch(node):
                yield step_ev

            blocks = _iter_content_blocks(token)
            tool_chunks = _tool_call_chunks_from_blocks(blocks)

            # Tool call streaming (from model node)
            if tool_chunks:
                # Ensure tool call started
                first = tool_chunks[0]
                chunk_id = first.get("id")
                chunk_name = first.get("name")

                if chunk_id and chunk_id != active_tool_call_id:
                    # Close previous tool call stream if any
                    if active_tool_call_id is not None:
                        yield ToolCallEndEvent(
                            tool_call_id=active_tool_call_id, timestamp=_now_ms()
                        )
                    active_tool_call_id = str(chunk_id)
                    active_tool_call_name = str(chunk_name) if chunk_name else "tool"
                    last_tool_call_id = active_tool_call_id

                    yield ToolCallStartEvent(
                        tool_call_id=active_tool_call_id,
                        tool_call_name=active_tool_call_name,
                        parent_message_id=message_id,
                        timestamp=_now_ms(),
                    )

                # Stream args deltas (including chunks with id=None/name=None but args fragments)
                if active_tool_call_id is not None:
                    for c in tool_chunks:
                        args = c.get("args")
                        if isinstance(args, str) and args:
                            yield ToolCallArgsEvent(
                                tool_call_id=active_tool_call_id,
                                delta=args,
                                timestamp=_now_ms(),
                            )

                continue

            # If we leave model node while a tool call is open, close it.
            if active_tool_call_id is not None and node and node != "model":
                yield ToolCallEndEvent(tool_call_id=active_tool_call_id, timestamp=_now_ms())
                active_tool_call_id = None
                active_tool_call_name = None

            # Tool node text becomes ToolCallResult (best-effort) and also may be shown as assistant text later.
            if node == "tools":
                tool_text = _text_from_blocks(blocks)
                if tool_text and last_tool_call_id:
                    yield ToolCallResultEvent(
                        message_id=new_id("toolmsg"),
                        tool_call_id=last_tool_call_id,
                        content=tool_text,
                        role="tool",
                        timestamp=_now_ms(),
                    )
                continue

            # Assistant text streaming (typically model node)
            delta = _text_from_blocks(blocks)
            if delta:
                if message_id is None:
                    message_id = new_id("msg")
                    yield TextMessageStartEvent(
                        message_id=message_id, role="assistant", timestamp=_now_ms()
                    )
                yield TextMessageContentEvent(
                    message_id=message_id, delta=delta, timestamp=_now_ms()
                )

        # Close any open tool call
        if active_tool_call_id is not None:
            yield ToolCallEndEvent(tool_call_id=active_tool_call_id, timestamp=_now_ms())

        # Close any open message stream
        if message_id is not None:
            yield TextMessageEndEvent(message_id=message_id, timestamp=_now_ms())

        if current_node is not None:
            yield StepFinishedEvent(step_name=current_node, timestamp=_now_ms())

        yield RunFinishedEvent(thread_id=thread_id, run_id=run_id, timestamp=_now_ms())
    except Exception as e:
        yield RunErrorEvent(message=str(e), code=type(e).__name__, timestamp=_now_ms())
        raise
