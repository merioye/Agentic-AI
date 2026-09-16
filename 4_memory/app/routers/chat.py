"""
Chat endpoint.

Design notes:
- user_id and thread_id are required and DISTINCT on purpose (see the
  memory-architecture pitfall about conflating them). thread_id is
  supplied by the client per-conversation; user_id should come from your
  auth layer in a real deployment, not from the request body - it's left
  as a body field here only to keep the tutorial runnable without wiring
  up auth.
- Errors from the LLM/tool layer are caught and translated into clean
  HTTP errors rather than leaking stack traces to the client.
- Episode recording triggers a BackgroundTask for procedural-memory
  reflection, kept off the request's critical path (see reflection.py's
  docstring for why).
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.reflection import reflect_and_update_playbook
from app.core.config import Settings
from app.core.deps import get_agent, get_settings_dep, get_store
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("support_agent.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


def _content_to_text(content: object) -> str:
    """Convert provider message content blocks into plain response text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Events frame. Two trailing newlines are
    what tell the browser's parser 'this event is complete' - get this
    wrong and event either merge together or never fires."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
    store=Depends(get_store),
    settings: Settings = Depends(get_settings_dep)
) -> ChatResponse:
    config = {
        "configurable": {
            "thread_id": payload.thread_id,
            "user_id": payload.user_id
        }
    }

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config,
        )
    except Exception:
        logger.exception("Agent invocation failed for user_id=%s thread_id=%s", payload.user_id, payload.thread_id)
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable.")

    final_message = result["messages"][-1]
    reply_text = _content_to_text(final_message.content)

    # Cheap heuristic trigger for this tutorial: If an episode tool was
    # called this turn, kick off a reflection pass in the background.
    tool_calls_made = any(
        getattr(m, "name", None) == "record_episode" for m in result["messages"]
    )
    if tool_calls_made:
        background_tasks.add_task(reflect_and_update_playbook, settings, store, payload.user_id)

    return ChatResponse(reply=reply_text, thread_id=payload.thread_id, user_id=payload.user_id)


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
    store=Depends(get_store),
    settings: Settings = Depends(get_settings_dep)
) -> StreamingResponse:
    """
    Same conversation turn as POST /chat, but streamed as Server-Sent
    Events so the UI can render tokens as they're generated instead of
    waiting for the full response.

    Event types emitted:
      token         - {"content": str}      a chunk of the reply text
      tool_start    - {"tool": str}         a memory tool was invoked
      tool_end      - {"tool": str}         that tool call finished
      error         - {"detail": str}       something went wrong mid-stream
      done          - {"thread_id": str}    stream is complete
    """
    config = {
        "configurable": {
            "thread_id": payload.thread_id,
            "user_id": payload.user_id,
        }
    }

    async def event_generator():
        episode_recorded = False
        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": payload.message}]},
                config,
                version="v2"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # AIMessageChunk.content can be "" for chunks that only
                    # carry tool-call deltas - skip those, nothing to render.
                    content = _content_to_text(chunk.content)
                    if content:
                        yield _sse("token", {"content": content})

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    yield _sse("tool_start", {"tool": tool_name})
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    if tool_name == "record_episode":
                        episode_recorded = True
                    yield _sse("tool_end", {"tool": tool_name})

                # Yield control back to the event loop between events so a
                # slow/long stream can't starve other concurrent requests.
                await asyncio.sleep(0)
        except Exception:
            logger.exception(
                "Streaming agent invocation failed for user_id=%s thread_id=%s",
                payload.user_id,
                payload.thread_id,
            )
            yield _sse("error", {"detail": "The assistant hit an error mid-response."})
            return

        if episode_recorded:
            background_tasks.add_task(reflect_and_update_playbook, settings, store, payload.user_id)

        yield _sse("done", {"thread_id": payload.thread_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disables buffering on nginx-frontend deployments - without this,
            # nginx can hold whole stream in a buffer and defeat the
            # purpose of streaming entirely.
            "X-Accel-Buffering": "no",
        }
    )