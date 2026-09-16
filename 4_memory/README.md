# Support Agent

A FastAPI + LangChain/LangGraph agent implementing all four CoALA memory
types: working, semantic, episodic, and procedural.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
uv sync
cp .env.example .env   # fill in your API keys
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/` for the browser chat UI (with streaming
and a live memory inspector panel), or `http://localhost:8000/docs` for
interactive OpenAPI docs.

## Architecture

| Memory type | Implementation                                                                                                                                                                      | Scope                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Working     | `InMemorySaver` / `PostgresSaver` (checkpointer)                                                                                                                                    | `thread_id` — one conversation            |
| Semantic    | LangGraph `Store`, namespace `("semantic_memories", user_id)`, via LangMem's `manage_memory`/`search_memory` tools (or Mem0, toggle via `MEMORY_BACKEND`)                           | `user_id` — all of a user's conversations |
| Episodic    | LangGraph `Store`, namespace `("episodic_memories", user_id)`, via custom typed `record_episode`/`recall_similar_episodes` tools                                                    | `user_id`                                 |
| Procedural  | LangGraph `Store`, namespace `("procedural_memory", "global")`, a single evolving playbook document injected into the system prompt each turn, updated by an offline reflection job | Global (all users), updated slowly        |

`thread_id` and `user_id` are always passed and used separately — never
conflate them. `thread_id` resets each conversation; `user_id` is the
durable identity long-term memory is keyed on.

## Endpoints

- `POST /chat` — send a message, get a reply in one shot (non-streaming). Body: `{user_id, thread_id, message}`.
- `POST /chat/stream` — same as above, but streams the reply as Server-Sent
  Events (`token`, `tool_start`, `tool_end`, `error`, `done` frames). Powers the browser UI.
- `GET /users/{user_id}/memories` — inspect a user's stored semantic + episodic memory.
- `DELETE /users/{user_id}/memories` — erase a user's stored memory (compliance/right-to-erasure).
- `GET /health` — liveness check.
- `GET /` — the browser chat UI (static file served from `app/static/index.html`).

## Browser UI

A single-file HTML/CSS/JS console at `app/static/index.html`, served
directly by FastAPI (no build step, no framework). Two panes:

- **Left**: chat, streamed token-by-token from `/chat/stream` via `fetch()`
  - a manually-parsed `ReadableStream` (plain `EventSource` can't be used
    here since it only supports GET requests without a body). Tool calls the
    agent makes mid-turn — `manage_memory`, `record_episode`, etc. — render
    as live chips so you can see memory writes happen in real time.
- **Right**: a live memory inspector that re-fetches
  `GET /users/{user_id}/memories` after each turn, split into semantic /
  episodic tabs, plus a button to erase the current user's memory.

Change the `user` field in the header to simulate a different person —
switching it and sending a message will show that semantic/episodic
memory is isolated per user, while switching `thread` alone (same user)
demonstrates that facts persist even in a brand-new conversation.

## Swapping to production persistence

Set in `.env`:

```
PERSISTENCE_BACKEND=postgres
POSTGRES_DSN=postgresql://...
```

## Swapping the long-term memory backend

Set `MEMORY_BACKEND=mem0` in `.env` to use Mem0 instead of LangMem's
store-based tools for semantic/episodic recall in the chat hot path.
Prefer Mem0 for latency-sensitive interactive agents; keep LangMem-style
reflection for offline procedural-memory updates regardless of which
backend you choose for recall (see `app/agents/reflection.py`).

## Known simplifications made for this tutorial

- `InMemoryStore`/`InMemorySaver` are the default — swap to Postgres before
  deploying anything real (see above).
- `user_id` is taken from the request body for simplicity. In production,
  derive it from your authentication layer, never trust a client-supplied
  identity for memory writes.
- The "episode recorded -> trigger reflection" heuristic in `chat.py` is
  intentionally simple; a real system would run reflection on a schedule
  over a batch of recent episodes rather than per-request.
