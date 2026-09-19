from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class ToolInfo(BaseModel):
    name: str
    description: str
    server: str


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]