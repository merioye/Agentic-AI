"""
FastAPI entry point.

    POST /api/expense/submit                start a new report, stream progress, may pause for approval
    POST /api/expense/resume                resume a paused report with a human decision, stream to completion
    GET /api/expense/history/{employee}     long-term memory: past decision from the Store
    GET /api/expense/{thread_id}/timeline   time-travel debug view over checkpointer history

Run with: uvicorn app.main:app --reload
"""
import json
import logging
import uuid
from typing import AsyncIterator, Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Durability

from app.config import ALLOWED_ORIGINS, DURABILITY_MODE, GRAPH_RECURSION_LIMIT
from app.graph import expense_graph
from app.schemas import ExpenseSubmitRequest, ResumeRequest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("expense_copilot")

app = FastAPI(title="Expense Approval Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_graph_run(
    graph_input: Any,
    config: RunnableConfig,
    *,
    durability: str = DURABILITY_MODE,
) -> AsyncIterator[str]:
    """Shared streaming logic for both a fresh run and resumed run.

    stream_mode=["custom", "updates"]: "custom" carries the progress events
    we emit ourselves via get_stream_writer(); "updates" carries per-node
    state deltas, and -- critically -- an "__interrupt__" chunk the instant
    a node calls interrupt(). We detect that chunk and stop streaming,
    telling the client a human decision is required.

    durability controls when checkpoints are flushed (Feature 6):
      "sync"  -- flush before the next node starts (safest, slowest)
      "async" -- flush in background while next node runs (default, balanced)
      "exit"  -- flush only when the graph exits (fastest, least safe)
    """
    try:
        async for stream_mode, chunk in expense_graph.astream(
            graph_input,
            config=config,
            stream_mode=["custom", "updates"],
            durability=cast(Durability, durability),
        ):
            if stream_mode == "updates":
                update_chunk = cast(dict, chunk)
                if "__interrupt__" in update_chunk:
                    interrupt_obj = update_chunk["__interrupt__"][0]
                    yield _sse({"type": "approval_required", **interrupt_obj.value})
                    return # graph is paused; wait for a /resume call
                node_name = next(iter(update_chunk), None)
                if node_name:
                    yield _sse({"type": "node_complete", "node": node_name})

            elif stream_mode == "custom":
                yield _sse({"type": "status", **cast(dict[str, Any], chunk)})


        # Graph ran to completion (either auto-approved, or resumed and
        # finalized) -- fetch the final persisted state for the result.
        snapshot = await expense_graph.aget_state(config)
        values = snapshot.values
        yield _sse({
            "type": "result",
            "decision": values.get("decision"),
            "decision_reason": values.get("decision_reason"),
            "total_amount": values.get("total_amount"),
            "flagged_items": [f for f in values.get("flagged_items", []) if f["requires_approval"]]
        })
        yield _sse({"type": "done"})

    except Exception:
        logger.exception("Error while streaming graph run for config %s", config)
        yield _sse({"type": "error", "message": "Something went wrong processing this report."})


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/expense/submit")
async def submit_expense(payload: ExpenseSubmitRequest) -> StreamingResponse:
    thread_id = str(uuid.uuid4())
    report_id = f"RPT-{thread_id[:8]}"

    line_items = [
        {
            "id": f"item-{i}",
            "category": item.category,
            "amount": item.amount,
            "description": item.description,
            "has_receipt": item.has_receipt
        }
        for i, item in enumerate(payload.line_items)
    ]

    graph_input = {
        "report_id": report_id,
        "employee": payload.employee,
        "line_items": line_items
    }
    config = cast(RunnableConfig, {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": int(GRAPH_RECURSION_LIMIT)
    })

    async def generator():
        yield _sse({"type": "session", "thread_id": thread_id, "report_id": report_id})
        async for frame in _stream_graph_run(graph_input, config):
            yield frame

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/expense/resume")
async def resume_expense(payload: ResumeRequest) -> StreamingResponse:
    config = cast(RunnableConfig, {
        "configurable": {"thread_id": payload.thread_id},
        "recursion_limit": int(GRAPH_RECURSION_LIMIT)
    })

    # Make sure this thread is actually paused before we try to resume it.
    snapshot = await expense_graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="This report is not waiting on approval.")

    resume_value = {"decision": payload.decision, "note": payload.note}

    async def generator():
        async for frame in _stream_graph_run(Command(resume=resume_value), config):
            yield frame

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )



@app.get("/api/expense/history/{employee}")
async def get_history(employee: str) -> dict:
    store = expense_graph.store
    assert store is not None
    items = store.search(("expense_history", employee), limit=25)
    return {
        "employee": employee,
        "items": [
            {"key": item.key, **item.value} for item in items
        ]
    }


@app.get("/api/expense/{thread_id}/timeline")
async def get_timeline(thread_id: str) -> dict:
    """Time-travel debug view: every checkpoint recorded for this thread,
    in order, with which node produced it. Demonstrates get_state_history()."""
    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    steps = []
    async for snapshot in expense_graph.aget_state_history(config):
        steps.append({
            "step": snapshot.metadata.get("step") if snapshot.metadata else None,
            "next_node": list(snapshot.next) if snapshot.next else None,
            "decision": snapshot.values.get("decision")
        })
    if not steps:
        raise HTTPException(status_code=404, detail="No checkpoint history for this thread)id.")
    steps.reverse() # oldest first
    return {"thread_id": thread_id, "steps": steps}


app.mount("/", StaticFiles(directory="static", html=True), name="static")