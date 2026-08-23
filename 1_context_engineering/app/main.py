"""
FastAPI entrypoint.

Run with:   uvicorn app.main:app --reload
Prod run:   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

Notes on the "many live users" target, addressed here and in README:
  - Agent + checkpointer are built ONCE at startup (lifespan) and reused
    across requests - never rebuild a LangChain agent per-request.
  - Runtime context (user identity) is constructed fresh per-request and
    never cached on the agent, so requests can't leak across users.
  - SQLite checkpointing is fine for a single worker process; for real
    horizontal scale (multiple uvicorn/gunicorn workers or pods), swap
    AsyncSqliteSaver for AsyncPostgresSaver - see docker-compose.yml and
    the note in app/agent.py. The rest of code is unchanged.
"""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import db
from app.agent import AgentContext, build_agent, run_turn, resume_turn, stream_turn
from app.auth import CurrentUser, get_current_user
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ResumeRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("support_agent")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db.init_db()
    logger.info("Building agent with model=%s", settings.primary_model)
    agent, checkpointer_cm = await build_agent()
    app.state.agent = agent
    app.state.checkpointer_cm = checkpointer_cm
    logger.info("Agent ready")
    yield
    await checkpointer_cm.__aexit__(None, None, None)
    logger.info("Shutdown complete")


app = FastAPI(title="Acme Support Agent", version="1.0.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=get_settings().primary_model)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: CurrentUser = Depends(get_current_user)):
    context = AgentContext(user_id=user.user_id, role=user.role)
    try:
        result = await run_turn(
            app.state.agent,
            thread_id=req.conversation_id,
            message=req.message,
            context=context
        )
    except Exception:
        logger.exception("Agent turn failed for conversation_id=%s", req.conversation_id)
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable.")
    return ChatResponse(**result)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, user: CurrentUser = Depends(get_current_user)):
    """Server-Sent Events endpoint. Streams the model's tokens
    as they're generated instead of blocking until the whole turn (including
    any tool calls) finishes. This is the single biggest perceived-latency
    win available: time-to-first-token drops from "the full turn" to
    roughly one model round-trip.

    Each SSE line is `data: <json>\\n\n` where the json matches on of the
    event shapes documented in app/agent.py's stream_turn().
    """
    context = AgentContext(user_id=user.user_id, role=user.role)

    async def event_source():
        try:
            async for event in stream_turn(
                app.state.agent,
                thread_id=req.conversation_id,
                message=req.message,
                context=context
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming turn failed for conversation_id=%s", req.conversation_id)
            error_event = {
                "type": "done",
                "message": "The assistant is temporarily unavailable.",
                "escalate_to_human": True,
                "requires_approval": False,
                "pending_action": None,
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/chat/resume", response_model=ChatResponse)
async def resume(req: ResumeRequest, user: CurrentUser = Depends(get_current_user)):
    if req.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
    
    context = AgentContext(user_id=user.user_id, role=user.role)
    try:
        result = await resume_turn(
            app.state.agent,
            thread_id=req.conversation_id,
            decision=req.decision,
            context=context
        )
    except Exception:
        logger.exception("Agent resume failed for conversation_id=%s", req.conversation_id)
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable.")
    return ChatResponse(**result)

# Serve the browser chat UI as static files at /ui
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")