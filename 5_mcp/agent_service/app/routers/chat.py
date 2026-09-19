"""
Chat endpoint, same SSE streaming pattern used in Day 4 project:
agent.astream_events() piped through StreamingResponse. The on
addition here is that tool_start/tool_end events now correspond to real
network calls into MCP servers - some local (our devops server spawned
as a subprocess), some potentially remote (the filesystem server, or in
a real deployment, Github/Slack/etc over HTTP). The UI can't tell the
difference, which is the point.
"""
from __future__ import annotations

import asyncio 
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.deps import get_agent, get_mcp_tools, get_tool_server_map
from app.schemas.chat import ChatRequest, ChatResponse, ToolInfo, ToolsResponse

logger = logging.getLogger("mcp_agent.chat")
router = APIRouter(tags=["chat"])

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


@router.get("/tools", response_model=ToolsResponse)
async def list_tools(tools=Depends(get_mcp_tools), server_map=Depends(get_tool_server_map)) -> ToolsResponse:
    return ToolsResponse(
        tools=[
            ToolInfo(
                name=t.name,
                description=(t.description or "").strip().split("\n")[0][:140],
                server=server_map.get(t.name, "unknown")
            )
            for t in tools
        ]
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, agent=Depends(get_agent)) -> ChatResponse:
    config = {"configurable": {"thread_id": payload.thread_id}}
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": payload.message}]}, config)
    except Exception:
        logger.exception("Agent invocation failed for thread_id=%s", payload.thread_id)
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable.")

    final_message = result["messages"][-1]
    reply_text = _content_to_text(final_message.content)
    return ChatResponse(reply=reply_text,thread_id=payload.thread_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    agent=Depends(get_agent),
    server_map=Depends(get_tool_server_map)
) -> StreamingResponse:
    config = {"configurable": {"thread_id": payload.thread_id}}

    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": payload.message}]},
                config,
                version="v2"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = _content_to_text(chunk.content)
                    if content:
                        yield _sse("token", {"content": content})
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    yield _sse("tool_start", {
                        "tool": tool_name,
                        "server": server_map.get(tool_name, "unknown"),
                        "input": event.get("data", {}).get("input", {})
                    })
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    output = event.get("data", {}).get("output")
                    yield _sse("tool_end", {
                        "tool": tool_name,
                        "server": server_map.get(tool_name, "unknown"),
                        "output": str(getattr(output, "content", output))[:300]
                    })
                elif kind == "on_tool_error":
                    yield _sse("tool_error", {
                        "tool": event.get("name", ""),
                        "detail": str(event.get("data", {}).get("error", "unknown error"))
                    })

                await asyncio.sleep(0)

        except Exception:
            logger.exception("Streaming agent invocation failed for thread_id=%s", payload.thread_id)
            yield _sse("error", {"detail": "The assistant hit an error mid-response."})
            return

        yield _sse("done", {"thread_id": payload.thread_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )