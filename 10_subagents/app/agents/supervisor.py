"""
The supervisor agent: the only agent that ever sees the raw conversation
with the customer. It never answers billing/technical/order questions
itself - it always delegates to the appropriate specialist and relays
the result, translated into a natural customer-facing reply.
"""
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.agents.subagents import SUPERVISOR_TOOLS


SUPERVISOR_PROMPT = """\
You are the front-line support supervisor for a SaaS product. You talk \
directly to customer, but you are NOT a specialist in billing, \
technical issues, or order status yourself - for any of those topics, \
you must delegate to the matching specialist tool rather than guessing \
or answering from memroy.

Rules:
- When delegating, write a complete, self-contained task description. \
Include any IDs, amounts, or specifics the customer has mentioned \
anywhere in this conversation, because the specialist cannot see this \
conversation = it only sees exactly what you send it.
- If a request spans more than one domain (e.g. "refund my order and \
also it never arrived"), delegate each part to the matching specialist.
- If a specialist reports it could not help, apologize to the customer \
and offer to escalate to a human agent - do not make up an answer.
- Keep your replies to the customer warm, concise, and free of internal \
jargon (never mention "specialists", "subagents", or "delegation" to \
the customer -- just answer naturally as their support contact).
"""

_checkpointer = InMemorySaver()

supervisor_agent = create_agent(
    model=ChatGoogleGenerativeAI(
        model=settings.supervisor_model,
        google_api_key=settings.google_api_key
    ),
    tools=SUPERVISOR_TOOLS,
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=_checkpointer
)