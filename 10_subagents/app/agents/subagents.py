"""
Subagents: eah is a full, independent create_agent() agent with its own
system prompt and its own narrow toolset. They are then wrapped as plain
async tools (`delegate_to_*`) so the supervisor an all them exactly like
any other tool.

Key production details in this file:
  - contextvars track which subagents were used per-request, safely under
    concurrent requests (a module-level list would leak across requests).
  - Every delegation is wrapped in try/except so a subagent failure becomes
    a clean message the supervisor an react to, not a crashed request.
  - recursion_limit caps each subagent's internal tool-call loop.
"""
import contextvars
import logging

from langchain.agents import create_agent
from langchain.tools import tool, ToolException
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.agents.tools import BILLING_TOOLS, TECHNICAL_TOOLS, ORDER_TOOLS
from app.schemas import message_content_to_text


logger = logging.getLogger("support_platform.subagents")


# Tracks subagent names used during the current request. contextvars (not a
# plain global) is what makes this safe: each async task gets its own
# isolated copy, so concurrent requests never see each other's data.
subagents_used: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "subagents_used", default=[]
)


def _record_usage(name: str) -> None:
    current = subagents_used.get()
    subagents_used.set(current + [name])


billing_agent = create_agent(
    model=ChatGoogleGenerativeAI(
        model=settings.subagent_model,
        google_api_key=settings.google_api_key
    ),
    tools=BILLING_TOOLS,
    system_prompt=(
        "You are a billing specialist for a SaaS company. You handle "
        "invoices, charges, and refunds. Be exact about dollar amounts "
        "and invoice IDs - never guess or round. If information you need "
        "wasn't provided (like a customer ID), say so plainly instead of "
        "making one up. You were reporting a result bak to an internal "
        "supervisor, not chatting with the end customer - keep your final "
        "answer short and factual."
    )
)


technical_agent = create_agent(
    model=ChatGoogleGenerativeAI(
        model=settings.subagent_model,
        google_api_key=settings.google_api_key
    ),
    tools=TECHNICAL_TOOLS,
    system_prompt=(
        "You are a technical support specialist. Check the known-issues "
        "knowledge base first. Only file an engineering ticket if the "
        "issue is not already a known, document issue. You are "
        "reporting a result back to an internal supervisor - keep your "
        "final answer short and factual, and state clearly whether you "
        "filed a ticket."
    )
)


orders_agent = create_agent(
    model=ChatGoogleGenerativeAI(
        model=settings.subagent_model,
        google_api_key=settings.google_api_key
    ),
    tools=ORDER_TOOLS,
    system_prompt=(
        "You are an order-status specialist. Look up orders by ID and "
        "report their status and ETA plainly. You are reporting a result "
        "back to an internal supervisor - keep your final answer short."
    )
)


async def _run_subagent(agent, name: str, task: str) -> str:
    """Shared execution path for every subagent: run it, log it, handle failure."""
    _record_usage(name)
    logger.info("subagent_call_start", extra={"subagent": name, "task": task})
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": settings.max_agent_steps}
        )
        answer = message_content_to_text(result["messages"][-1].content)
        logger.info("subagent_call_end", extra={"subagent": name, "success": True})
        return answer
    except Exception:
        logger.exception("subagent_call_failed", extra={"subagent": name})
        raise ToolException(
            f"The {name} specialist is temporarily unavailable. "
            "Apologize to the customer and offer to escalate to a human agent."
        )


@tool
async def delegate_to_billing(task: str) -> str:
    """
    Delegate a billing-related task to the billing specialist subagent.
    Use for anything involving invoices, charges, refunds, or payments.
    The task description must be self-contained - include any specific
    customer IDs, invoice IDs, or dollar amounts from the conversation,
    since the specialist cannot see the original conversation.
    """
    return await _run_subagent(billing_agent, "billing", task)


@tool
async def delegate_to_technical(task: str) -> str:
    """
    Delegate a technical-support task to the technical specialist subagent.
    Use for bugs, errors, login problems, sync issues, or anything requiring
    troubleshooting. The task description must be self-contained.
    """
    return await _run_subagent(technical_agent, "technical", task)


@tool
async def delegate_to_orders(task: str) -> str:
    """
    Delegate an order-status task to the orders specialist subagent.
    Use for shipping status, delivery ETAs, or order tracking questions.
    The task description must be self-contained and include the order ID
    if the customer provided one.
    """
    return await _run_subagent(orders_agent, "orders", task)


SUPERVISOR_TOOLS = [delegate_to_billing, delegate_to_technical, delegate_to_orders]