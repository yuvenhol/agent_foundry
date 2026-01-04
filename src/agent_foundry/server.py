"""FastAPI server exposing AG-UI SSE endpoints for MasterAgent and Runtime."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core.events import CustomEvent
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from .agui.adapter import langchain_astream_events_to_agui_events
from .agui.encoding import sse_encode_event
from .agui.ids import new_id
from .master_agent import Context, MasterAgent
from .runtime import Runtime
from .schemas import AgentSpec, CommonChatReq
from .tools.registry import tool_registry


def _agui_messages_to_langchain(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert AG-UI-style message dicts to LangChain message objects."""
    converted: list[Any] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            # Unsupported/unknown roles are passed through as user text for now.
            converted.append(HumanMessage(content=str(content)))
    return converted


def create_app() -> FastAPI:
    app = FastAPI(title="agent_foundry (AG-UI)")

    async def _master_stream_response(req: CommonChatReq) -> StreamingResponse:
        thread_id = req.sessionId or new_id("thread")
        run_id = req.message.id or new_id("run")

        master = await MasterAgent.build()
        context = Context()

        lc_messages = [HumanMessage(content=req.message.content)]

        async def stream() -> AsyncIterator[str]:
            async for event in langchain_astream_events_to_agui_events(
                runnable=master,
                input={"messages": lc_messages},
                config={"configurable": {"context": context}},
                thread_id=thread_id,
                run_id=run_id,
            ):
                yield sse_encode_event(event)

            if context.agent_spec is not None:

                yield sse_encode_event(
                    CustomEvent(
                        name="agent_spec",
                        value=context.agent_spec.model_dump(),
                    )
                )

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def _runtime_stream_response(payload: dict[str, Any]) -> StreamingResponse:
        thread_id = payload.get("thread_id") or new_id("thread")
        run_id = payload.get("run_id") or new_id("run")
        messages = payload.get("messages") or []
        spec_raw = payload.get("agent_spec")
        if spec_raw is None:
            raise HTTPException(status_code=422, detail="agent_spec is required")

        try:
            spec = AgentSpec(**spec_raw)
        except ValidationError as e:
            # Return structured validation errors instead of 500 so the frontend can surface it.
            raise HTTPException(status_code=422, detail=e.errors()) from e
        available_tools = tool_registry.get_all()
        runnable = await Runtime.build(agent_spec=spec, all_available_tools=available_tools)

        lc_messages = _agui_messages_to_langchain(messages)

        async def stream() -> AsyncIterator[str]:
            async for event in langchain_astream_events_to_agui_events(
                runnable=runnable,
                input={"messages": lc_messages},
                config=None,
                thread_id=thread_id,
                run_id=run_id,
            ):
                yield sse_encode_event(event)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/agui/master/run")
    async def agui_master_run(req: CommonChatReq) -> StreamingResponse:
        return await _master_stream_response(req)

    @app.post("/agui/runtime/run")
    async def agui_runtime_run(payload: dict[str, Any]) -> StreamingResponse:
        return await _runtime_stream_response(payload)

    @app.get("/agui/runtime/run")
    async def agui_runtime_run_get(payload: str = Query(...)) -> StreamingResponse:
        """GET variant for EventSource clients (payload is URL-encoded JSON)."""
        try:
            decoded = json.loads(payload)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid payload JSON: {e}") from e
        if not isinstance(decoded, dict):
            raise HTTPException(status_code=422, detail="payload must be a JSON object")
        return await _runtime_stream_response(decoded)

    return app
