"""
Memory inspection and deletion endpoints.

Any system that stores long-term memory about real users needs a
deletion path from day on - not bolted on later. This is a common
production and compliance requirement (GDPR "right to erasure" and
similar), and it's easy to forget when memory is scattered across
multiple namespaces or backends.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.deps import get_store
from app.core.memory import episodic_namespace, semantic_namespace
from app.schemas.chat import MemoryEntry, UserMemoriesResponse

router = APIRouter(prefix="/users", tags=["memory-admin"])


@router.get("/{user_id}/memories", response_model=UserMemoriesResponse)
async def list_user_memories(user_id: str, store=Depends(get_store)) -> UserMemoriesResponse:
    semantic_items = store.search(semantic_namespace(user_id), query="", limit=100)
    episodic_items = store.search(episodic_namespace(user_id), query="", limit=100)

    return UserMemoriesResponse(
        user_id=user_id,
        semantic=[MemoryEntry(key=r.key, value=r.value) for r in semantic_items],
        episodic=[MemoryEntry(key=r.key, value=r.value) for r in episodic_items]
    )


@router.delete("/{user_id}/memories", status_code=204)
async def delete_user_memories(user_id: str, store=Depends(get_store)) -> None:
    for namespace in (semantic_namespace(user_id), episodic_namespace(user_id)):
        for item in store.search(namespace, query="", limit=1000):
            store.delete(namespace, item.key)