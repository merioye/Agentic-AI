"""
The heavyweight planning pattern: planner -> executor -> replanner, as a
genuine loop rather than a single ReAct-style agent.

IMPLEMENTATION NOTE: conceptually this is exactly the LangGraph
plan-and-execute tutorial's node structure (planner/executor/replanner
with a conditional edge back to executor or END) - but it's implemented
here as a direct orchestrator (`_run_loop`) rather than a compiled
`StateGraph`. Reason: a compiled StateGraph has one fixed entry point, and
human-approval flow needs TWO entry points - "plan a fresh goal" and
"execute an already-approved (possible human-edited) plan, skipping the
planner call entirely." Driving the same node functions from plain Python
sidesteps that constraint with no loss of the conceptual structure; swap
in a real `StateGraph` (nodes unchanged) if you need LangGraph-native
features here (checkpointed interrups, visual graph tracing) that
this project doesn't require.
"""
from __future__ import annotations

import asyncio
from typing import Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools import EXECUTOR_TOOLS

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PlanStepSchema(BaseModel):
    id: str = Field(description="Short id, e.g. 's1', 's2'")
    description: str = Field(description="Exactly what this step should accomplish")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of steps that must complete before this one can start"
    )


class PlanSchema(BaseModel):
    steps: list[PlanStepSchema]


class ReplanDecision(BaseModel):
    done: bool = Field(description="True if the goal is now fully achieved")
    final_message: str | None = Field(default=None, description="Required if done=true: the final QBR summary")
    add_steps: list[PlanStepSchema] = Field(
        default_factory=list,
        description="New steps to add ONLY if results genuinely require plan revision - leave empty otherwise"
    )


class StepResult(BaseModel):
    summary: str = Field(description="Concise result of completing this step, including concrete numbers found")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
_planner_model = None

def _get_planner():
    global _planner_model
    if _planner_model is None:
        settings = get_settings()
        model = ChatGoogleGenerativeAI(
            model=settings.primary_model,
            google_api_key=settings.google_api_key
        )
        _planner_model = model.with_structured_output(PlanSchema)
    return _planner_model


_PLANNER_PROMPT = """Break this goal into a plan of concrete steps for a \
team of data-gathering/calculation tools to execute. Each step needs a \
short id and a clear description of exactly what to do. Mark dependencies \
via depends_on (ids of steps that must complete first) - steps with NO \
dependency relationship to each other should be left independent tso they \
can run in parallel; don't add unnecessary dependencies. Include a final \
synthesis step that depends on all the data-gathering steps, to combine \
findings into recommendations.

Goal: {goal}"""


async def planner_node(state: dict) -> dict:
    settings = get_settings()
    planner = _get_planner()
    plan_schema = cast(
        PlanSchema,
        await planner.ainvoke(
            _PLANNER_PROMPT.format(goal=state["goal"])
        )
    )
    steps = [
        {"id": s.id, "description": s.description, "depends_on": s.depends_on, "status": "pending", "result": None}
        for s in plan_schema.steps[:settings.max_plan_steps]
    ]
    return {
        "plan": steps,
        "report_sections": [],
        "replan_cycles": 0,
        "done": False,
        "final_message": None
    }


# ---------------------------------------------------------------------------
# Executor - runs every step whose dependencies are satisfied CONCURRENTLY
# Each step is executed by a small, cheap sub-agent.
# ---------------------------------------------------------------------------
_executor_agent = None


async def _get_executor_agent():
    global _executor_agent
    if _executor_agent is None:
        settings = get_settings()
        model = ChatGoogleGenerativeAI(
            model=settings.executer_model,
            google_api_key=settings.google_api_key
        )
        _executor_agent = create_agent(
            model=model,
            tools=EXECUTOR_TOOLS,
            middleware=[ModelCallLimitMiddleware(run_limit=5)],
            response_format=StepResult
        )
    return _executor_agent


async def _execute_step(step: dict, dependency_results: dict[str, str]) -> str:
    agent = await _get_executor_agent()
    context_lines = [f"- {dep_id}: {res}" for dep_id, res in dependency_results.items()]
    context_block = ("\n\nResults from steps this depends on:\n" + "\n".join(context_lines)) if context_lines else ""
    prompt = f"Complete this specific task: {step['description']}{context_block}"
    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    reply: StepResult = result["structured_response"]
    return reply.summary


def _get_ready_steps(plan: list[dict]) -> list[dict]:
    """Pure DAG-scheduling logic: every step still presuending whose
    dependencies have ALL completed. Extracted as its own function
    so it's directly testable without needing a live agent."""
    completed_ids = {s["id"] for s in plan if s["status"] == "completed"}
    return [s for s in plan if s["status"] == "pending" and set(s["depends_on"]).issubset(completed_ids)]


async def executor_node(state: dict) -> dict:
    plan = state["plan"]
    ready = _get_ready_steps(plan)
    if not ready:
        return {} # nothing runnable this cycle

    async def run_one(step: dict) -> tuple[str, str, str]:
        dep_results = {
            dep_id: next((s["result"] for s in plan if s["id"] == dep_id), "")
            for dep_id in step["depends_on"]
        }
        try:
            summary = await _execute_step(step, dep_results)
            return step["id"], "completed", summary
        except Exception as e: # noqa BLE001 - a failed step is data, not a crash
            return step["id"], "failed", f"Step failed: {e}"

    # THE concrete payoff of explicit dependencies: every ready
    # step executes concurrently, not one at a time.
    results = await asyncio.gather(*[run_one(s) for s in ready])
    result_map = {sid: (status, summary) for sid, status, summary in results}

    updated_plan = []
    new_sections = []
    for s in plan:
        if s["id"] in result_map:
            status, summary = result_map[s["id"]]
            s = {**s, "status": status, "result": summary}
            if status == "completed":
                new_sections.append({"title": s["description"], "content": summary})
        updated_plan.append(s)

    return {"plan": updated_plan, "report_sections": state.get("report_sections", []) + new_sections}


# ---------------------------------------------------------------------------
# Replanner - decides retry-in-place / revise-the-plan / done
# ---------------------------------------------------------------------------
_replanner_model = None


def _get_replanner():
    global _replanner_model
    if _replanner_model is None:
        model = ChatGoogleGenerativeAI(
            model=get_settings().primary_model,
            google_api_key=get_settings().google_api_key
        )
        _replanner_model = model.with_structured_output(ReplanDecision)
    return _replanner_model

_REPLAN_PROMPT = """Goal: {goal}

Current plan status:
{plan_status}

Report sections gathered so far: {sections}

Decide: is the goal now fully achieved (done=true, with a final_message \
that is the actual QBR summary and recommendations), or does the plan \
need revising with new steps (done=false, add_steps populated)? Only add \
steps if something in the results genuinely requires it - do not add \
redundant steps just because you can."""


async def replanner_node(state: dict) -> dict:
    settings = get_settings()
    cycles = state.get("replan_cycles", 0) + 1
    if cycles > settings.max_replan_cycles:
        return {
            "done": True,
            "final_message": "Reached the maximum replanning cycles - stopping with partial results.",
            "replan_cycles": cycles
        }

    plan = state["plan"]
    all_settled = all(s["status"] in ("completed", "failed") for s in plan)
    if not all_settled:
        # Steps are still pending/running - nothing to decide yet, just
        # let the executor keep going. No LLM call needed for this case.
        return {"replan_cycles": cycles}

    plan_status = "\n".join(
        f"- [{s['status']}] {s['id']}: {s['description']} -> {s.get('result') or '(no result)'}" for s in plan
    )
    sections = ", ".join(sec["title"] for sec in state.get("report_sections", [])) or "(none yet)"

    replanner = _get_replanner()
    decision: ReplanDecision = cast(
        ReplanDecision, 
        await replanner.ainvoke(
            _REPLAN_PROMPT.format(
                goal=state["goal"],
                plan_status=plan_status,
                sections=sections,
            )
        )
    )

    if decision.done:
        return {"done": True, "final_message": decision.final_message, "replan_cycles": cycles}

    new_steps = [
        {"id": s.id, "description": s.description, "depends_on": s.depends_on, "status": "pending", "result": None}
        for s in decision.add_steps
    ]
    if not new_steps:
        # Replanner said "not done" but added nothing runnable - without
        # this guard the loop would spin forever with no ready steps.
        return {
            "done": True,
            "final_message": decision.final_message or "Plan complete - no further steps identified.",
            "replan_cycles": cycles,
        }

    return {"plan": plan + new_steps, "replan_cycles": cycles}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def should_continue(state: dict) -> Literal["executor", "end"]:
    return "end" if state.get("done") else "executor"


async def _run_loop(state: dict) -> dict:
    while True:
        state = {**state, **(await executor_node(state))}
        state = {**state, **(await replanner_node(state))}
        if should_continue(state) == "end":
            return state


async def create_plan(goal: str) -> dict:
    """Plan-only call, no execution, for human review."""
    update = await planner_node({"goal": goal})
    return {"goal": goal, **update}


async def run_plan(goal: str, plan: list[dict] | None = None) -> dict:
    """IF `plan` is provided (e.g. returned by create_plan and possibly
    edited by a human reviewer), execution starts directly from it,
    skipping the planner call entirely - the human-approval flow.
    Otherwise plans fresh from the goal."""
    if plan is not None:
        state = {"goal": goal, "plan": plan, "report_sections": [], "replan_cycles": 0, "done": False, "final_message": None}
    else:
        state = {"goal": goal, **(await planner_node({"goal": goal}))}
    final_state = await _run_loop(state)
    return final_state


