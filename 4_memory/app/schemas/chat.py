from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable identity across all of this user's conversations - powers long-term memory.")
    thread_id: str = Field(..., min_length=1, description="Scopes this specific conversation - powers working memory.")
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    user_id: str


class MemoryEntry(BaseModel):
    key: str
    value: dict


class UserMemoriesResponse(BaseModel):
    user_id: str
    semantic: list[MemoryEntry]
    episodic: list[MemoryEntry]