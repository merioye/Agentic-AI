"""
The agent. Effort is chosen BEFORE the agent runs (app/main.py classifies
the incoming question via app/complexity_router.py and puts the result in
AgentContext) rather than inside a middleware hook - this keeps
apply_reasoning_effort a plain synchronous wrap_model_call, the same shape
as every prior middleware, and treats "how hard should we think about this"
as a request-level decision, which is what is actually is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, cast
from collections.abc import AsyncIterator, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    PIIMiddleware,
    dynamic_prompt,
    wrap_model_call,
    AgentMiddleware
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.reasoning import extract_reasoning, model_for_effort
from app.tools import ALL_TOOLS


@dataclass
class AgentContext:
    user_id: str | None
    conversation_id: str
    effort: str # one of config.EFFORT_LEVELS - set per-request by app/main.py


class ReasoningReply(BaseModel):
    message: str = Field(description="The final answer")
    escalate_to_human: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Model Context middleware
# ---------------------------------------------------------------------------
@dynamic_prompt
def reasoning_prompt(request: ModelRequest) -> str:
    return (
        "You are a quantitative analysis assistant. You have calculator "
        "tool and a reference-data lookup tool.\n"
        "- Always use lookup_reference_data before reasoning about any "
        "specific metric or threshold - never guess a figure.\n"
        "- Always use evaluate_expression for arithmetic rather than doing "
        "math yourself, to guarantee accuracy.\n"
        "- After a tool result, reconsider your plan before deciding the "
        "next step - don't blindly execute a fixed sequence if the result "
        "changes what's actually needed.\n"
        "- Give clear, direct answers with the specific numbers involved."
    )


@wrap_model_call
async def apply_reasoning_effort(
    request: ModelRequest[AgentContext],
    handler: Callable[[ModelRequest[AgentContext]], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Swap in the model selected for this request's reasoning effort."""
    effort = request.runtime.context.effort
    request = request.override(model=model_for_effort(effort))

    return await handler(request)


# ---------------------------------------------------------------------------
# Agent factory - no checkpointer/store needed; each request is evaluated
# independently (a reasoning benchmark/analysis tool, not a long-running
# support conversation) so in-memory-only state is a deliberate
# simplification, not an oversight.
# ---------------------------------------------------------------------------
async def build_agent():
    settings = get_settings()

    primary_model = ChatGoogleGenerativeAI(
        model=settings.primary_model,
        google_api_key=settings.google_api_key
    )

    middleware = cast(
        list[AgentMiddleware[Any, AgentContext, Any]],
        [
            reasoning_prompt,
            apply_reasoning_effort,
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
            ),
            ModelCallLimitMiddleware(
                run_limit=settings.max_model_calls_per_turn,
            ),
        ],
    )

    agent = create_agent(
        model=primary_model,
        tools=ALL_TOOLS,
        middleware=middleware,
        context_schema=AgentContext,
        response_format=ReasoningReply
    )

    return agent


# ---------------------------------------------------------------------------
# Turn helpers
# ---------------------------------------------------------------------------
async def run_turn(agent, *, message: str, context: AgentContext) -> dict:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        context=context
    )
    reply: ReasoningReply = result["structured_response"]
    reasoning_summary, thinking_tokens = extract_reasoning(result["messages"])
    return {
        "message": reply.message,
        "escalate_to_human": reply.escalate_to_human,
        "effort_used": context.effort,
        "reasoning_summary": reasoning_summary,
        "thinking_tokens": thinking_tokens
    }


async def stream_turn(agent, *, message: str, context: AgentContext) -> AsyncIterator[dict[str, Any]]:
    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        context=context, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            token, metadata = chunk
            if metadata.get("langgraph_node") == "model":
                text = getattr(token, "context", "") or ""
                if text:
                    yield {"type": "token", "text": text}
        elif mode == "updates":
            for node_name in chunk:
                if node_name == "tools":
                    yield {"type": "tool_end", "tool": "analysis"}

    # Streaming doesn't hand back a final state object the same way
    # ainvoke does, and this project has not checkpointer to re-fetch state
    # from - so the streamed path reports effort_used but not the post-hoc
    # thinking-token/summary breakdown. Use POST /chat (non-streaming) when
    # you need those numbers
    yield {
        "type": "done", "effort_used": context.effort,
        "reasoning_summary": None, "thinking_tokens": None
    }