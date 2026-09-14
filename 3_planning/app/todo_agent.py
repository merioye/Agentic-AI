"""
The lightweight planning pattern: one middleware `TodoListMiddleware`,
added to an otherwise-normal create_agent. The model plans and re-plans
in-context via a `write_todos` tool the middleware injects - no separate
planner/executor graph, still one continuous agent loop.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    dynamic_prompt,
    AgentMiddleware
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from app.config import get_settings
from app.tools import ALL_TOOLS


class TodoAgentReply(BaseModel):
    message: str = Field(description="Summary of the completed QBR, referencing the recorded report sections")
    escalate_to_human: bool = Field(default=False)


class QBRAgentState(AgentState):
    # report_sections comes from app/tools.py's write_report_section tool,
    # NOT from TodoListMiddleware - that middleware separately contributes
    # its own `todos` field to agent state. create_state merges every
    # middleware's state schema with this one automatically.
    report_sections: NotRequired[Annotated[list[dict], operator.add]]


@dynamic_prompt
def qbr_prompt(request) -> str:
    return (
        "You are producing a Quarterly Business Review (QBR) covering "
        "Marketing, Sales, Engineering Cloud Spend, and Support Ticket "
        "Volume.\n"
        "Use write_todos to plan your approach before diving in, and keep "
        "it updated as you make progress (mark items in_progress/completed "
        "as you go - don't just write the list once and ignore it).\n"
        "For EACH department: look up its data, calculate the Q2->Q3 "
        "percentage change, and note whether Q3 crosses its alert "
        "threshold. Record each department's findings with "
        "write_report_sections as you finish reasoning about it.\n"
        "Finish with a synthesis section (also via write_report_section) "
        "recommending next steps for any department at or near its threshold."
    )


async def build_agent():
    settings = get_settings()

    model = ChatGoogleGenerativeAI(
        model=settings.primary_model,
        google_api_key=settings.google_api_key
    )

    summarizer_model = ChatGoogleGenerativeAI(
        model=settings.executer_model,
        google_api_key=settings.google_api_key
    )

    middleware = cast(
        list[AgentMiddleware[Any, Any, Any]],
        [
            qbr_prompt,
            TodoListMiddleware(),
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            SummarizationMiddleware(model=summarizer_model, trigger={"tokens": 4000}, keep=("messages", 20)),
            ModelCallLimitMiddleware(run_limit=settings.max_model_calls_per_turn)
        ],
    )

    agent = create_agent(
        model=model,
        tools=ALL_TOOLS,
        middleware=middleware,
        state_schema=QBRAgentState,
        response_format=TodoAgentReply
    )

    return agent


async def run_goal(agent, *, goal: str) -> dict:
    result = await agent.ainvoke({"messages": [{"role": "user", "content": goal}]})
    reply: TodoAgentReply = result["structured_response"]
    todos = result.get("todos", []) or []
    return {
        "message": reply.message,
        "escalate_to_human": reply.escalate_to_human,
        "report_sections": result.get("report_sections", []) or [],
        "todos": todos,
        "todos_completed": sum(1 for t in todos if t.get("status") == "completed"),
        "todos_total": len(todos)
    }