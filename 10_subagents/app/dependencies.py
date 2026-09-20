"""
FastAPI dependencies.

A "dependency" in FastAPI is just a function FastAPI calls before your
route handler runs, and whose return value (or raised exception) it
injects. It's how you do auth, DB sessions, and shared setup without
repeating that code in every endpoint.
"""
from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Require a valid API key on every protected request, passed as the
    `X-API-Key` header. FastAPI's `Header(...)` means this parameter is
    required — a request missing it is rejected with a 422 before this
    function even runs.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
