"""
FastAPI entrypoint for the AI support platform.

Run with:
    uvicorn app.main:app --reload

Then POST to /support with header `X-API-Key`: <your key>`.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain.tools import ToolException

from app.config import settings
from app.logging_config import configure_logging
from app.schemas import SupportRequest, SupportResponse, ErrorResponse, message_content_to_text
from app.dependencies import verify_api_key
from app.agents.supervisor import supervisor_agent
from app.agents.subagents import subagents_used

configure_logging()
logger = logging.getLogger("support_platform.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: wire up optional LangSmith tracing from settings rather than
    # requiring the caller to set raw env vars themselves.
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("langsmith_tracing_enabled", extra={"project": settings.langsmith_project})
    logger.info("app_startup")
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="AI Support Platform",
    description="Supervisor + subagents customer support API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all so an unexpected error never leaks a raw stack trace to a
    client. The real error is logged server-side with full detail; the
    client gets a safe, generic message.
    """
    logger.exception("unhandled_exception", extra={"path": str(request.url)})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="Something went wrong. Please try again shortly."
        ).model_dump()
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/support",
    response_model=SupportResponse,
    dependencies=[Depends(verify_api_key)],
    responses={401: {"model": ErrorResponse}, 504: {"model": ErrorResponse}}
)
async def handle_support_request(payload: SupportRequest) -> SupportResponse:
    """
    Main entrypoint: takes a customer message, runs it through the
    supervisor agent (which may delegate to on or more subagents),
    and returns the final reply.
    """
    # Reset the per-request subagent-usage tracker. Because FastAPI/Starlette
    # runs each request in its own asyncio Task (which gets its own copy of 
    # the context), this is scoped to this request even without extra work
    # we reset explicitly anyway for clarity and to guard against reuse.
    subagents_used.set([])

    logger.info(
        "support_request_received",
        extra={"customer_id": payload.customer_id, "session_id": payload.session_id}
    )

    try:
        result = await asyncio.wait_for(
            supervisor_agent.ainvoke(
                {"messages": [{"role": "user", "content": payload.message}]},
                config={
                    "configurable": {"thread_id": payload.session_id},
                    "recursion_limit": settings.max_agent_steps
                }
            ),
            timeout=settings.request_timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error("support_request_timeout", extra={"session_id": payload.session_id})
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The request took tool long to process. Please try again."
        )
    except ToolException as exc:
        # A subagent raised a clean, user-safe message - pass it through
        # as the reply rather than treating it as a hard failure.
        logger.warning("support_request_tool_exception", extra={"session_id": payload.session_id})
        return SupportResponse(
            session_id=payload.session_id,
            reply=str(exc),
            subagents_used=subagents_used.get()
        )

    # Capture usage immediately after the call succeeds - read this before
    # anything else has a chance to reset or overwrite it.
    used = subagents_used.get()
    reply = message_content_to_text(result["messages"][-1].content)

    logger.info(
        "support_request_completed",
        extra={"session_id": payload.session_id, "subagents_used": used}
    )

    return SupportResponse(session_id=payload.session_id, reply=reply, subagents_used=used)


_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    