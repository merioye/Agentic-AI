"""
FastAPI entrypoint.

Run with:   uvicorn app.main:app --reload

Endpoints:
  POST /todo/chat               Run the goal through the TodoListMiddleware agent
  POST /plan-execute/plan       Plan-only - no execution, for human review
  POST /plan-execute/run        Execute a goal, optionally starting from an approved plan
  GET /health

No auth in this project - same scoping decision as Day 2, no
access-control-relevant data here.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import plan_execute, todo_agent
from app.config import get_settings
from app.schemas import (
    GoalRequest,
    HealthResponse,
    PlanResponse,
    RunPlanRequest,
    RunPlanResponse,
    TodoChatResponse
)


# Set LangSmith env vars before any langchain imports
settings = get_settings()

os.environ["LANGSMITH_TRACING_V2"] = str(settings.langsmith_tracing_v2).lower()
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("planning_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Building todo-list agent with model=%s", settings.primary_model)
    app.state.todo_agent = await todo_agent.build_agent()
    logger.info("Agents ready")
    yield
    logger.info("Shutdown complete")


app = FastAPI(title="Planning & Goal Decomposition", version="1.0.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=get_settings().primary_model)


@app.post("/todo/chat", response_model=TodoChatResponse)
async def todo_chat(req: GoalRequest):
    try:
        result = await todo_agent.run_goal(app.state.todo_agent, goal=req.goal)
    except Exception:
        logger.exception("Todo-agent run failed for goal=%r", req.goal)
        raise HTTPException(status_code=502, detail="The agent is temporarily unavailable.")
    return TodoChatResponse(**result)


@app.post("/plan-execute/plan", response_model=PlanResponse)
async def plan_only(req: GoalRequest):
    try:
        result = await plan_execute.create_plan(req.goal)
    except Exception:
        logger.exception("Planning failed for goal=%r", req.goal)
        raise HTTPException(status_code=502, detail="Planning is temporarily unavailable.")
    return PlanResponse(goal=result["goal"], plan=result["plan"])


@app.post("/plan-execute/run", response_model=RunPlanResponse)
async def plan_and_run(req: RunPlanRequest):
    plan = [step.model_dump() for step in req.plan] if req.plan is not None else None
    try:
        result = await plan_execute.run_plan(req.goal, plan=plan)
    except Exception:
        logger.exception("Plan-execute run failed for goal=%r", req.goal)
        raise HTTPException(status_code=502, detail="Execution is temporarily unavailable.")
    return RunPlanResponse(
        goal=req.goal,
        final_message=result.get("final_message"),
        plan=result.get("plan", []),
        report_sections=result.get("report_sections", []),
        replan_cycles=result.get("replan_cycles", 0)
    )


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")