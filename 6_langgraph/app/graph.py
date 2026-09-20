"""
Expense approval pipeline -- the LangGraph mastery project.

Topology:

    START   ->  intake  ->  [Send fan-out: one review item per line item, parallel]
            ->  review_item (many, in parallel) ->  aggregate
            ->  route_after_aggregate   ->  human_approval  OR  auto_approve
            ->  finalize    -> END

This exercises in one realistic pipeline:
    - Send()                dynamic map-reduce over a variable number of line items
    - reducers              flagged_items accumulates across all parallel review_item runs
    - interrupt()/Command   pausing for a human approver and resuming later, possibly from a completely different HTTP request
    - checkpointer          thread-scoped persistence so the pause survives across requests
    - Store                 cross-thread long-term memory of past approval decisions
    - custom streaming      progress events the UI can render live

Production-grade resilience features added:
    - RetryPolicy           exponential back-off retries on the LLM call inside review_item
    - TimeoutPolicy         hard wall-clock cap per LLM attempt (composes with retries)
    - graph error handler   graceful fallback written to state when a node exhausts retries
    - @task idempotency     LLM call is a durable task; checkpoint-backed replays skip re-execution
    - CachePolicy           identical line items served from cache within a TTL window
    - Durability            configurable checkpoint flush mode (sync / async / exit)
"""
import logging
from typing import Literal, TypedDict, Annotated, Any, cast
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import (
    CachePolicy,
    RetryPolicy,
    Send,
    TimeoutPolicy,
    interrupt,
)
from pydantic import BaseModel, Field

from app.config import (
    APPROVAL_TOTAL_THRESHOLD,
    DEFAULT_MODEL,
    GOOGLE_API_KEY,
    LLM_RETRY_INITIAL_INTERVAL,
    LLM_RETRY_MAX_ATTEMPTS,
    REVIEW_ITEM_TIMEOUT,
)

logger = logging.getLogger("expense_copilot")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class LineItem(TypedDict):
    id: str
    category: str
    amount: float
    description: str
    has_receipt: bool


class FlaggedItem(TypedDict):
    item_id: str
    category: str
    amount: float
    requires_approval: bool
    reason: str


class ExpenseState(TypedDict):
    report_id: str
    employee: str
    line_items: list[LineItem]
    flagged_items: Annotated[list[FlaggedItem], operator.add]
    total_amount: float
    decision: str | None
    decision_reason: str | None


class _ItemReviewInput(TypedDict):
    """The custom, per-branch input Send() hands to review_item. Deliberately
    NOT the full ExpenseState -- each parallel branch only needs one item."""
    report_id: str
    employee: str
    item: LineItem


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

class ItemAssessment(BaseModel):
    requires_approval: bool = Field(
        description="True if a human should review this line item before it's approved"
    )
    reason: str = Field(
        description="One short sentence: why it's flagged, or 'Within policy' if fine"
    )


_model = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=GOOGLE_API_KEY
)
_structured_model = _model.with_structured_output(ItemAssessment)

POLICY_PROMPT = (
    "You are an expense-policy reviewer. Flag a line item (requires_approval=True) if ANY of: "
    "it has no receipt attached, it is in the 'entertainment' or 'travel' category and over $150, "
    "the description doesn't plausibly match the stated category, or the amount looks unusually "
    "high for that category. Otherwise requires_approval=False with reason 'Within policy'."
)


# ---------------------------------------------------------------------------
# Resilience policies
# ---------------------------------------------------------------------------

# Retry: exponential back-off on LLM failures.
# initial_interval doubles each attempt up to max_interval, with jitter to
# avoid thundering-herd. Only applied to _call_llm_for_item (the one node
# that calls an external service); all other nodes are pure computation.
REVIEW_RETRY = RetryPolicy(
    initial_interval=LLM_RETRY_INITIAL_INTERVAL,  # e.g. 1 s
    backoff_factor=2.0,
    max_interval=30.0,
    max_attempts=LLM_RETRY_MAX_ATTEMPTS,           # e.g. 3 attempts total
    jitter=True,
)

# Timeout: hard wall-clock cap per attempt.
# Composes with retries: NodeTimeoutError triggers the retry policy.
# Timeout is only supported for async tasks (review_item is async, so safe).
REVIEW_TIMEOUT = TimeoutPolicy(run_timeout=REVIEW_ITEM_TIMEOUT)  # e.g. 30 s

# Cache: skip the LLM for identical line items within the TTL.
# The default key function hashes the full task input (all LineItem fields),
# so the same (category, amount, description, has_receipt) tuple is a hit.
REVIEW_CACHE = CachePolicy(ttl=600)  # 10-minute TTL


# ---------------------------------------------------------------------------
#  @task: idempotent LLM call
#
# Wrapping the LLM invocation in @task gives us:
#   - Durability: when a checkpointer is active the task result is persisted
#     the instant it completes. Replaying the graph (e.g. after a crash mid
#     fan-out) skips the LLM call and reads the result from the checkpoint.
#   - Retries & timeout live on the task, not on add_node, because the task
#     is the granular unit of work that can fail and be retried independently.
#   - Cache: the cache_policy here plugs into the InMemoryCache wired at
#     compile() time; identical inputs are served without hitting the model.
# ---------------------------------------------------------------------------

@task(
    retry_policy=REVIEW_RETRY,
    timeout=REVIEW_TIMEOUT,
    cache_policy=REVIEW_CACHE,
)
async def _call_llm_for_item(item: LineItem) -> ItemAssessment:
    """Durable, retriable, cached LLM assessment for one line item.

    Because this is a @task, its result is checkpointed the instant it
    completes. A graph replay (e.g. after a server restart mid-fan-out)
    will skip re-invoking the model and read from the checkpoint instead.
    """
    res = await _structured_model.ainvoke([
        {"role": "system", "content": POLICY_PROMPT},
        {
            "role": "user",
            "content": (
                f"Category: {item['category']}, Amount: ${item['amount']:.2f}, "
                f"Description: {item['description']}, Has receipt: {item['has_receipt']}"
            )
        }
    ])
    return cast(ItemAssessment, res)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def intake(state: ExpenseState) -> dict:
    total = sum(item["amount"] for item in state["line_items"])
    return {"total_amount": round(total, 2), "flagged_items": []}


def fan_out_to_reviews(state: ExpenseState) -> list[Send]:
    """Dynamic map: one Send per line item, however many there are."""
    return [
        Send(
            "review_item",
            {"report_id": state["report_id"], "employee": state["employee"], "item": item}
        )
        for item in state["line_items"]
    ]


async def review_item(state: _ItemReviewInput) -> dict:
    """Review a single line item against expense policy.

    The expensive LLM call is delegated to the @task-decorated
    _call_llm_for_item, which provides idempotency, retries, timeout, and
    caching. This node is a thin orchestration wrapper that streams progress
    events and assembles the flagged_items output.
    """
    writer = get_stream_writer()
    item = state["item"]
    writer({"event": "reviewing_item", "item_id": item["id"], "category": item["category"]})

    # Dispatch the @task and await its Future. If the LLM call fails,
    # REVIEW_RETRY fires automatically before the exception surfaces here.
    # On a graph replay, this line returns immediately from the checkpoint.
    future = _call_llm_for_item(item)
    assessment: ItemAssessment = await future

    writer({
        "event": "item_reviewed",
        "item_id": item["id"],
        "requires_approval": assessment.requires_approval,
        "reason": assessment.reason,
    })

    return {
        "flagged_items": [
            {
                "item_id": item["id"],
                "category": item["category"],
                "amount": item["amount"],
                "requires_approval": assessment.requires_approval,
                "reason": assessment.reason
            }
        ]
    }


def aggregate(state: ExpenseState) -> dict:
    """Fan-in point: every review_item run has an edge to here, so this runs
    exactly once, after all parallel branches complete, with flagged_items
    fully accumulated by the reducer."""
    writer = get_stream_writer()
    n_flagged = sum(1 for f in state["flagged_items"] if f["requires_approval"])
    writer({"event": "aggregated", "total_amount": state["total_amount"], "flags": n_flagged})
    return {}


def route_after_aggregate(state: ExpenseState) -> Literal["human_approval", "auto_approve"]:
    any_flagged = any(f["requires_approval"] for f in state["flagged_items"])
    if any_flagged or state["total_amount"] > float(APPROVAL_TOTAL_THRESHOLD):
        return "human_approval"
    return "auto_approve"


def auto_approval(state: ExpenseState) -> dict:
    return {
        "decision": "approved",
        "decision_reason": "Auto-approved: no flagged items and total under threshold."
    }


def human_approval(state: ExpenseState) -> dict:
    # Everything below this line re-runs on resume -- it's pure recomputation
    # from state, so it's safe to repeat. Nothing here calls an external
    # system or has a side effect, which is exactly the constraint interrupt()
    # imposes on the code that precedes it.
    flagged = [f for f in state["flagged_items"] if f["requires_approval"]]

    decision = interrupt({
        "report_id": state["report_id"],
        "employee": state["employee"],
        "total_amount": state["total_amount"],
        "flagged_items": flagged,
        "message": "This report needs manager approval before it can be finalized."
    })

    return {
        "decision": decision.get("decision", "rejected"),
        "decision_reason": decision.get("note", "")
    }


def finalize(state: ExpenseState, *, store: BaseStore) -> dict:
    # Long-term memory: record this outcome per employee/category so a future
    # dashboard (or a smarter routing node) could reference approval history.
    # Note this is cross-thread -- it outlives this run's thread_id entirely.
    namespace = ("expense_history", state["employee"])
    for item in state["line_items"]:
        store.put(
            namespace,
            f"{state['report_id']}:{item['id']}",
            {
                "category": item["category"],
                "amount": item["amount"],
                "decision": state["decision"],
                "report_id": state["report_id"]
            }
        )
    return {}


# ---------------------------------------------------------------------------
# Graph-level error handler
#
# Registered via set_graph_defaults(error_handler=...) so it fires whenever
# any regular node raises an exception that exhausts its retries (or has no
# retry policy). The handler receives the current state and writes a clean
# "error" decision, letting the graph finalize gracefully instead of
# propagating an unhandled exception to the HTTP streaming layer.
#
# The error handler is NOT retried and does NOT go through normal edge routing.
# ---------------------------------------------------------------------------

def graph_error_handler(state: ExpenseState) -> dict:
    """Fallback invoked when a node fails after all retry attempts.

    Records a graceful 'error' outcome in state so the SSE stream emits a
    proper 'result' event with decision='error' instead of a raw exception.
    """
    logger.error(
        "Graph error handler fired for report_id=%s employee=%s",
        state.get("report_id", "unknown"),
        state.get("employee", "unknown"),
    )
    return {
        "decision": "error",
        "decision_reason": (
            "Processing failed after retries. Please resubmit the report. "
            "If the problem persists, contact support."
        ),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ExpenseState)

    # register the error handler before compile() so it covers
    # every node in the graph. Per-node overrides could be passed to
    # add_node(..., error_handler=...) but a single graph-wide fallback is
    # the right default for this pipeline.
    graph.set_node_defaults(error_handler=graph_error_handler)

    graph.add_node("intake", intake)

    # review_item is now an async node (it awaits the @task future).
    # Retry / timeout / cache policies live on the @task decorator above,
    # not here -- the task is the granular unit of work that can fail.
    graph.add_node("review_item", review_item)

    graph.add_node("aggregate", aggregate)
    graph.add_node("auto_approve", auto_approval)
    graph.add_node("human_approval", human_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", fan_out_to_reviews, ["review_item"])
    graph.add_edge("review_item", "aggregate")
    graph.add_conditional_edges(
        "aggregate", route_after_aggregate, ["human_approval", "auto_approve"]
    )
    graph.add_edge("human_approval", "finalize")
    graph.add_edge("auto_approve", "finalize")
    graph.add_edge("finalize", END)

    # InMemorySaver / InMemoryStore / InMemoryCache are fine for this demo.
    # In production, swap for PostgresSaver / PostgresStore + a Redis-backed
    # cache so paused approvals, history, and cache entries survive restarts.
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    # Feature 5: InMemoryCache backend wired at compile() time.
    # The REVIEW_CACHE policy on _call_llm_for_item uses this backend to
    # serve identical line-item assessments without hitting the LLM.
    cache = InMemoryCache()

    return graph.compile(checkpointer=checkpointer, store=store, cache=cache)


expense_graph = build_graph()

