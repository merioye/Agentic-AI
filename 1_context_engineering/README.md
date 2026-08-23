# Acme Support Agent — Day 1 Reference Application

A complete, runnable implementation of the **context engineering with LangChain**
FastAPI backend + LangChain v1.x agent + a CLI client + a browser UI,
all talking to the same API.

```
1_context_engineering/
├── app/
│   ├── main.py        FastAPI app (lifespan, /chat, /chat/resume, /health)
│   ├── agent.py        LangChain agent: tools, middleware, context engineering
│   ├── auth.py          API-key auth dependency (demo — see notes inside)
│   ├── config.py        pydantic-settings configuration
│   ├── db.py             SQLite "orders" datastore (stand-in for a real service)
│   └── schemas.py        Request/response models
├── ui/index.html         Browser chat console (no build step — plain HTML/JS)
├── cli.py                  Rich-based terminal chat client
├── pyproject.toml
└── .env.example
```

## What this demonstrates

Every context-engineering concept is here as working code, not
pseudocode:

| Concept               | Where                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| Runtime Context       | `AgentContext` in `agent.py`, built fresh per-request in `main.py` from the authenticated user     |
| Dynamic system prompt | `support_prompt` (`@dynamic_prompt`)                                                               |
| Dynamic tool gating   | `gate_tools_by_role` (`@wrap_model_call`) — guests get **zero** order tools                        |
| Tool Context (read)   | `list_my_orders`, `get_order_status` read `runtime.context`                                        |
| Tool Context (write)  | `process_refund` writes to the orders DB                                                           |
| Structured output     | `SupportReply` — the API never parses free text                                                    |
| Life-cycle middleware | `PIIMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `HumanInTheLoopMiddleware` |
| Persistence           | `AsyncSqliteSaver` checkpointer keyed by `thread_id` = `conversation_id`                           |

## 1. Setup

```bash
cd 1_context_engineering
uv sync

cp .env.example .env
# edit .env and set GOOGLE_API_KEY (or OPENAI_API_KEY + PRIMARY_MODEL=openai:...)
```

## 2. Run the API

```bash
uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Browser UI: http://localhost:8000/ui/
- Interactive API docs: http://localhost:8000/docs

## 3. Talk to it

**Browser:** open http://localhost:8000/ui/, pick an identity from the
dropdown (`guest-key`, `customer-key-1`, `customer-key-2`, `admin-key`), and chat.
Try:

- As `guest-key`: "What's the status of my order?" → agent explains you must log in (tool-gated, not just prompt-instructed).
- As `customer-key-1`: "What orders do I have?" then "Refund ORD-1001, it arrived broken" → triggers a human-approval step in the UI.

**CLI:**

```bash
python cli.py --api-key customer-key-1
```

**Raw HTTP:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: customer-key-1" \
  -d '{"conversation_id": "demo-1", "message": "What orders do I have?"}'
```

---

## Scaling notes: getting to "thousands of live users"

This app runs correctly as-is for a single-process demo. Here's what
changes as load grows, in the order you'd actually hit them:

1. **Multiple workers, one machine.** `uvicorn --workers N` (or the
   Dockerfile's `--workers 2`) gets you multi-core concurrency. At this
   stage, **SQLite checkpointing breaks** — multiple processes writing to
   one SQLite file will lock/contend. Swap `AsyncSqliteSaver` for
   `AsyncPostgresSaver` (`langgraph-checkpoint-postgres`) pointed at the
   Postgres service in `docker-compose.yml`. The rest of `agent.py` is
   unchanged — the checkpointer is a drop-in interface.
2. **Multiple machines / pods.** Once the checkpointer is Postgres-backed,
   the FastAPI app itself is stateless and horizontally scalable — run as
   many replicas as you want behind a load balancer. Conversation state
   lives in Postgres, keyed by `thread_id`, not in process memory.
3. **Long-term memory (`store`) at scale.** `InMemoryStore` is
   single-process only — replace with a Postgres- or Redis-backed store
   before scaling past one process, same reasoning as the checkpointer.
4. **Rate limiting & backpressure.** Add per-identity rate limiting (e.g.
   `slowapi`, or push this to an API gateway / reverse proxy) — LLM calls
   are slow and expensive relative to a normal HTTP handler; an
   unthrottled endpoint is a cost and availability risk.
5. **Timeouts and circuit breaking.** `ModelCallLimitMiddleware` already
   caps runaway loops per conversation turn; pair this with a
   request-level timeout (`request_timeout_seconds` in config) enforced at
   your reverse proxy, and `ModelFallbackMiddleware` for provider outages.
6. **Observability.** Wire `LANGCHAIN_TRACING_V2=true` + a LangSmith API
   key to get per-call traces in production, and emit structured
   (JSON) logs — the current `logging.basicConfig` call is a placeholder;
   swap for `structlog` or your platform's logging convention.
7. **Auth.** Replace `app/auth.py`'s static API-key dict with real
   OAuth2/JWT verification against your identity provider — the rest of
   the app depends only on receiving a `CurrentUser`, so nothing else
   changes.
8. **Cost control.** `PIIMiddleware` + `SummarizationMiddleware` already
   bound token growth per conversation; add per-tenant usage tracking
   (e.g. writing token counts to your `store` or a metrics backend) if
   you're billing or capping usage per customer.

None of these require re-architecting the agent logic in `agent.py` — that's
the point of building on LangGraph's checkpointer/store abstractions instead
of hand-rolling persistence.
