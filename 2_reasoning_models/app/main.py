"""
FastAPI entrypoint.

Run with: uvicorn app.main:app --reload

Endpoints:
  POST /chat            non-streaming turn - reports effort/reasoning/cost
  POST /chat/stream     SSE streaming turn
  GET /health

No auth in this project - there's no access control relevant data here, so
the usual auth scaffolding would be noise, not a real production concern
for what this app does.
"""
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import EFFORT_LEVELS, get_settings
from app.complexity_router import classify_effort
from app.reasoning_agent import AgentContext, build_agent, run_turn, stream_turn
from app.schemas import ChatRequest, ChatResponse, HealthResponse

# Set LangSmith env vars before any langchain imports
settings = get_settings()

os.environ["LANGSMITH_TRACING_V2"] = str(settings.langsmith_tracing_v2).lower()
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("reasoning_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Building reasoning agent with model=%s", settings.primary_model)
    app.state.agent = await build_agent()
    logger.info("Agent ready")
    yield
    logger.info("Shutdown complete")


app = FastAPI(title="Reasoning Effort Router", version="1.0.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=get_settings().primary_model)


async def _resolve_effort(req: ChatRequest) -> str:
    if req.force_effort:
        if req.force_effort not in EFFORT_LEVELS:
            raise HTTPException(status_code=400, detail=f"force_effort must be one of {EFFORT_LEVELS}")
        return req.force_effort
    return await classify_effort(req.message)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    effort = await _resolve_effort(req)
    context = AgentContext(user_id=None, conversation_id=req.conversation_id, effort=effort)
    try:
        result = await run_turn(app.state.agent, message=req.message, context=context)
    except Exception:
        logger.exception("Agent turn failed for conversation_id=%s", req.conversation_id)
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable.")
    return ChatResponse(**result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    effort = await _resolve_effort(req)
    context = AgentContext(user_id=None, conversation_id=req.conversation_id, effort=effort)

    async def event_source():
        yield f"data: {json.dumps({'type': 'effort_selected', 'effort': effort})}\n\n"
        try:
            async for event in stream_turn(app.state.agent, message=req.message, context=context):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming turn failed for conversation_id=%s", req.conversation_id)
            error_event = {"type": "done", "message": "The assistant is temporarily unavailable.", "effort_used": effort}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_source(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")