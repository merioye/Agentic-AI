"""
The agent itself - this is "Day 1: Context Engineering":

  - AgentContext            -> Runtime Context (static, per-request config)
  - support_prompt          -> dynamic_prompt middleware (Model Context)
  - gate_tools_by_role      -> wrap_model_call middleware (Model Context)
  - get_order_status /
    process_refund /
    list_my_orders          -> tools that read Runtime Context (Tool Context)
  - SummarizationMiddleware, PIIMiddleware, ModelCallLimitMiddleware,
    HumanInTheLoopMiddleware -> Life-cycle Context

  WRITE     -> SupportAgentState.notes + save_note tool: a scratchpad the
               agent can write to that survives summarization (state, not
               a message), explicitly re-injected into the prompt every
               turn so it can't be silently dropped.
  ISOLATE   -> SupportAgentState.order_cache: list_my_orders stores full
               order data in state (isolated from the LLM's token budget)
               and returns only a short summary + count. Full detail only
               enters context when the model asks about a specific order_id.
  COMPRESS  -> summarize_after_tokens lowered to fight "context distraction"
               - trim well before the window is actually full, not at 95%.
  GROUNDING -> prompt language tightened: explicit "never state an order
               number, price, or status from memory" rule, and tool
               docstrings tightened so the model can't rationalize
               skipping them.
  SPEED     -> tools are `async def` (parallelizable by the tool-
               calling step), and stream_turn() gives token-level streaming
               instead of blocking on the full ainvoke().
"""
from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, AsyncIterator
from typing import Any, cast, Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    PIIMiddleware,
    SummarizationMiddleware,
    dynamic_prompt,
    wrap_model_call,
    AgentState,
    AgentMiddleware
)
from langchain.tools import ToolRuntime, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from app import db
from app.config import get_settings

# ---------------------------------------------------------------------------
# Runtime Context: static, per-request configuration the agent can read from.
# Built fresh on every API call from authenticated user - never cached
# on the agent object, never shared across requests.
# --------------------------------------------------------------------------
@dataclass
class AgentContext:
    user_id: str | None
    role: str # "guest" | "customer" | "admin"

# --------------------------------------------------------------------------
# Custom agent state - state_schema must be a TypedDict in LangChain
# We extend the built-in AgentState
#
#   notes           -> WRITE: agent-authored scratchpad. Annotated with
#                      operator.add so Command(update={"notes": [...]}) APPENDS
#                      rather than overwrites - same reducer pattern "messages"
#                      uses internally. Survives summarization because it's a
#                      state field, not a message in the trimmed list.
#  order_cache      -> ISOLATE: full tool output kept out of the LLM's context;
#                      only a compact summary is shown unless asked for detail.
#                      No reducer needed - each list_my_orders call replaces it
#                      with the current full set.
# --------------------------------------------------------------------------
class SupportAgentState(AgentState):
    notes: NotRequired[Annotated[list[str], operator.add]]
    order_cache: NotRequired[dict[str, dict]]


# --------------------------------------------------------------------------
# Structured output: the agent's final reply is coerced into this shape,
# so the FastAPI layer never has to regex-parse free text.
# --------------------------------------------------------------------------
class SupportReply(BaseModel):
    """Final structured reply returned to the API client."""

    message: str = Field(description="Reply shown to the user, plain text")
    escalate_to_human: bool = Field(description="True if the conversation should be handed to a human agent")


# --------------------------------------------------------------------------
# Tools - these both read Runtime Context (user identity) and touch the
# "orders" datastore. process_refund is gated by HumanInTheLoopMiddleware
# below, so the LLM can propose it but never execute it unsupervised.
# --------------------------------------------------------------------------
@tool
async def list_my_orders(runtime: ToolRuntime[AgentContext]) -> Command | str:
    """List all orders belonging to the current authenticated user.
    Returns a short summary — call get_order_status with a specific order_id
    for full detail on one order rather than assuming details from this list."""
    if runtime.context.role == "guest" or not runtime.context.user_id:
        return "You need to be logged in to view orders."
    orders = await asyncio.to_thread(db.list_orders, runtime.context.user_id)
    if not orders:
        return "No orders found."

    # ISOLATE: cache full records in state; only a compact line goes to the LLM.
    cache = {o["order_id"]: o for o in orders}
    summary = ", ".join(f"{o['order_id']} ({o['status']})" for o in orders)
    return Command(
        update={
            "order_cache": cache,
            "messages": [
                {
                    "role": "tool",
                    "content": f"{len(orders)} orders(s): {summary}",
                    "tool_call_id": runtime.tool_call_id
                }
            ]
        }
    )


@tool
async def get_order_status(order_id: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Look up full status details for ONE specific order by its order ID.
    Always call this before stating any order's status, price, or refund
    state to the user — never answer from memory or from a prior summary."""
    if runtime.context.role == "guest" or not runtime.context.user_id:
        return "You need to be logged in to check order status."
    order = await asyncio.to_thread(db.get_order, order_id, runtime.context.user_id)
    if not order:
        return f"No order found with ID {order_id} for this account."
    return (
        f"Order {order['order_id']}: {order['item']}, status={order['status']}, "
        f"amount=${order['amount']:.2f}, refunded={bool(order['refunded'])}"
    )


@tool
async def process_refund(order_id: str, reason: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Issue a refund for a delivered or shipped order. Sensitive action —
    requires human approval before it actually executes (see HumanInTheLoopMiddleware).
    Always call get_order_status first to confirm the order exists and its
    current refunded state before proposing this."""
    if not runtime.context.user_id:
        return f"User ID is required to process refund order ID {order_id}."
    order = await asyncio.to_thread(db.get_order, order_id, runtime.context.user_id)
    if not order:
        return f"No order found with ID {order_id} for this account."
    if order["refunded"]:
        return f"Order {order_id} was already refunded."
    await asyncio.to_thread(db.mark_refunded, order_id, runtime.context.user_id)
    return f"Refund processed for order {order_id} (${order['amount']:.2f}). Reason: {reason}"

@tool
async def save_note(note: str, runtime: ToolRuntime[AgentContext]) -> Command:
    """WRITE a short note to your scratchpad - use this to record something
    you'll need later in a long conversation (e.g. 'customer's issue is a
    damaged item, not a change of mind') so it survives even if earlier
    messages get summarized away. Do not use this to store fats you should
    instead be looking up with a tool."""
    return Command(
        update={
            "notes": [note], # appended via operator.add reducer on state
            "messages": [
                {
                    "role": "tool",
                    "content": f"Noted: {note}",
                    "tool_call_id": runtime.tool_call_id
                }
            ]
        }
    )
ALL_TOOLS = [list_my_orders, get_order_status, process_refund, save_note]

# --------------------------------------------------------------------------
# Model Context middleware
# --------------------------------------------------------------------------
@dynamic_prompt
def support_prompt(request: ModelRequest[AgentContext]) -> str:
    context = request.runtime.context

    if context is None:
        return (
            "You are performing an internal conversation summarization task. "
            "Summarize the provided conversation accurately and concisely. "
            "Do not make tool calls or invent information."
        )

    role = context.role
    notes = request.state.get("notes") or []

    base = (
        "You are Acme Retail's customer support agent. Be concise, warm, and factual.\n"
        "Grounding rules (do not violate these):\n"
        "- NEVER state an order number, price, status, or refund outcome from memory "
        "or from an earlier summary. Always call get_order_status for the specific "
        "order first, even if you already mentioned it earlier in the conversation.\n"
        "- If a tool returns 'no order found', say so plainly — do not guess or "
        "invent a plausible-looking order.\n"
        "- Use save_note for things worth remembering mid-conversation (e.g. the "
        "customer's stated reason for contacting support), not for order facts."
    )
    if role == "guest":
        base += (
            "\nThe current user is NOT logged in. You cannot access any order data. "
            "Politely ask them to log in with their account."
        )
    elif role == "admin":
        base += "\nThe current user is a support admin and may process refunds directly."

    if notes:
        base += "\n\nScratchpad notes from earlier in this conversation:\n" + "\n".join(
            f"- {n}" for n in notes
        )
    return base

@wrap_model_call
async def gate_tools_by_role(
    request: ModelRequest[AgentContext],
    handler: Callable[
        [ModelRequest[AgentContext]],
        Awaitable[ModelResponse],
    ],
) -> ModelResponse:
    """Guests get no order tools at all — enforced at the tool-list level,
    not just via prompt instructions, so the model literally cannot call
    something it wasn't given."""
    context = request.runtime.context

    # Internal model calls may not have AgentContext.
    if context is None:
        return await handler(request)
    
    role = request.runtime.context.role
    if role == "guest":
        request = request.override(tools=[])
    return await handler(request)


# --------------------------------------------------------------------------
# Agent factory
# --------------------------------------------------------------------------
async def build_agent():
    settings = get_settings()

    checkpointer_cm = AsyncSqliteSaver.from_conn_string(settings.sqlite_checkpoint_path)
    checkpointer = await checkpointer_cm.__aenter__() # kept open for app lifetime
    store = InMemoryStore() # swap for a Postgres-backed store at production scale

    primary_model = ChatGoogleGenerativeAI(
        model=settings.primary_model,
        google_api_key=settings.google_api_key,
    )
    summarizer_model = ChatGoogleGenerativeAI(
        model=settings.summarizer_model,
        google_api_key=settings.google_api_key
    )

    model_limit_middleware = cast(
        AgentMiddleware[AgentState, AgentContext, Any],
        ModelCallLimitMiddleware(
            run_limit=settings.max_model_calls_per_turn,
        ),
    )

    agent = create_agent(
        model=primary_model,
        tools=ALL_TOOLS,
        middleware=[
            support_prompt,
            gate_tools_by_role,
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            SummarizationMiddleware(
                model=summarizer_model,
                trigger={"tokens": settings.summarize_after_tokens},
                keep=("messages", 20)
            ),
            model_limit_middleware,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "process_refund": {"allowed_decisions": ["approve", "reject"]}
                }
            )
        ],
        context_schema=AgentContext,
        state_schema=SupportAgentState,
        response_format=SupportReply,
        checkpointer=checkpointer,
        store=store
    )

    return agent, checkpointer_cm


# --------------------------------------------------------------------------
# Thin invoke/resume helpers used by the FastAPI layer. Kept here so both
# the HTTP API and the CLI client share identical agent-calling logic.
#  --------------------------------------------------------------------------
def _extract_interrupt(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    payload = interrupts[0].value
    action_requests = payload.get("actionRequests") or payload.get("action_requests") or []
    description = None
    if action_requests:
        first = action_requests[0]
        description = f"{first.get('name')}({first.get('args')})"
    return {"description": description}

def _state_interrupt(state) -> dict | None:
    """Interrupt introspection via agent.aget_state(), used by stream_turn
    since a plain astream doesn't surface __interrupt__ the way ainvoke
    does. NOTE: LangGraph's exact interrupt-inspection surface has shifted
    across releases - if this breaks against installed version, the
    canonical source of truth is `state.tasks[i].interrupts`; adjust the
    field access below to match installed langgraph version's docs."""
    for task in getattr(state, "tasks", []) or []:
        task_interrupts = getattr(task, "interrupts", None)
        if task_interrupts:
            payload = task_interrupts[0].value
            action_requests = payload.get("actionRequests") or payload.get("action_requests") or []
            description = None
            if action_requests:
                first = action_requests[0]
                description = f"{first.get('name')}({first.get('args')})"
            return {"description": description}
        return None


async def run_turn(agent, *, thread_id: str, message: str, context: AgentContext) -> dict:
    """Non-streaming turn — kept for callers (tests, simple integrations)
    that just want the final answer in one shot."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        context=context,
    )
    interrupt = _extract_interrupt(result)
    if interrupt:
        return {
            "message": "This action needs approval before I can continue.",
            "escalate_to_human": False,
            "requires_approval": True,
            "pending_action": interrupt["description"]
        }
    reply: SupportReply = result["structured_response"]
    return {
        "message": reply.message,
        "escalate_to_human": reply.escalate_to_human,
        "requires_approval": False,
        "pending_action": None
    }


async def resume_turn(agent, *, thread_id: str, decision: str, context: AgentContext) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        Command(resume={"decisions": [{"type": decision}]}),
        config=config,
        context=context
    )
    interrupt = _extract_interrupt(result)
    if interrupt:
        return {
            "message": "Another action needs approval before I can continue.",
            "escalate_to_human": False,
            "requires_approval": True,
            "pending_action": interrupt["description"]
        }
    reply: SupportReply = result["structured_response"]
    return {
        "message": reply.message,
        "escalate_to_human": reply.escalate_to_human,
        "requires_approval": False,
        "pending_action": None
    }

async def stream_turn(agent, *, thread_id: str, message: str, context: AgentContext) -> AsyncIterator[dict[str, Any]]:
    """Token-level streaming turn. Yields small event dicts:
        {"type": "token", "text": "..."}            model output, as generated
        {"type": "tool_end", "tool": "..."}         a tool-calling step finished
        {"type": "done", ...ChatResponse fields}    terminal event, always last
    Consumed by app/main.py's SSE endpoint and by cli.py / ui/index.html.
    """
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        context=context,
        stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            token, metadata = chunk
            if metadata.get("langgraph_node") == "model":
                text = getattr(token, "content", "") or ""
                if text:
                    yield {"type": "token", "text": text}
        elif mode == "updates":
            for node_name in chunk:
                if node_name == "tools":
                    yield {"type": "tool_end", "tool": "tool_call"}

    # The stream above ends either because the turn completed normally, or
    # because HumanInTheLoopMiddleware paused execution. Check final state
    # to tell which happened.
    state = await agent.aget_state(config)
    interrupt = _state_interrupt(state)
    if interrupt:
        yield {
            "type": "done",
            "message": "This action needs approval before I can continue.",
            "escalate_to_human": False,
            "requires_approval": True,
            "pending_action": interrupt["description"]
        }
        return

    reply: SupportReply | None = state.values.get("structured_response")
    if reply is None:
        yield {
            "type": "done",
            "message": "Sorry, something went wrong generating a response.",
            "escalate_to_human": True,
            "requires_approval": False,
            "pending_action": None,
        }
        return

    yield {
        "type": "done",
        "message": reply.message,
        "escalate_to_human": reply.escalate_to_human,
        "requires_approval": False,
        "pending_action": None
    }