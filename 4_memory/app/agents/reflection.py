"""
Procedural memory update job.

This is intentionally NOT called inline during a chat request. Reflection
over multiple episodes to propose playbook changes i exactly the kind of
heavier, reasoning-intensive operation that belongs in a background task
or a scheduled job - running it in the request/response hot path is the
#1 way teams accidentally ship a chat endpoint with tens-of-seconds
latency spikes.

In this project it's triggered via FastAPI BackgroundTasks after a hat
turn that records an episode (see app/routers/chat.py), and in a real
deployment you'd typically run it as a nightly/hourly scheduled job over
newly recorded episodes instead of per-request.
"""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.base import BaseStore

from app.core.config import Settings
from app.core.memory import (
    PROCEDURAL_NAMESPACE,
    PROCEDURAL_KEY,
    episodic_namespace,
    get_playbook
)

REFLECTION_PROMPT = """You maintain the operating playbook for a customer
support agent. Below is the current playbook and a batch of recently
resolved support episodes.

Current playbook:
{playbook}

Recent episodes:
{episodes}

Propose an updated playbook as a short bullet list of imperative rules.
Only add or change a rule if a clear, repeated pattern in the episodes
justifies it - do not invent rules form a single anecdote. Keep the total
list under 10 rules. Return ONLY the bullet list, nothing else."""


def reflect_and_update_playbook(settings: Settings, store: BaseStore, user_id: str) -> None:
    recent = store.search(episodic_namespace(user_id), query="", limit=20)
    if not recent:
        return

    episodes_text = "\n\n".join(
        f"- Situation: {r.value['situation']}\n"
        f"  Action taken: {r.value['action_taken']}\n"
        f"  Outcome: {r.value['outcome']}"
        for r in recent
    )

    reflector = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key
    )
    prompt = REFLECTION_PROMPT.format(playbook=get_playbook(store), episodes=episodes_text)
    proposal = reflector.invoke(prompt).content

    store.put(PROCEDURAL_NAMESPACE, PROCEDURAL_KEY, {"rules": proposal})