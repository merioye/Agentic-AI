# AI support platform — supervisor + subagents

A FastAPI service backed by a LangChain supervisor agent that delegates
to three specialist subagents (billing, technical support, order status).

## Architecture

```
Customer --> FastAPI /support --> Supervisor agent --> delegate_to_billing    --> Billing subagent    --> billing tools
                                                     --> delegate_to_technical --> Technical subagent  --> technical tools
                                                     --> delegate_to_orders    --> Orders subagent      --> order tools
```

The supervisor is the only agent that ever sees the customer's message.
Each subagent is a fully independent `create_agent()` agent with its own
system prompt and narrow toolset, invoked through a thin async wrapper
tool. Only the subagent's final answer returns to the supervisor — its
internal tool calls and reasoning never enter the supervisor's context.

## Project layout

```
app/
  config.py            # pydantic-settings config, loaded from .env
  logging_config.py     # structured JSON logging
  schemas.py            # request/response Pydantic models
  dependencies.py        # API key auth dependency
  main.py                # FastAPI app + /support endpoint + static UI mount
  agents/
    tools.py             # domain tools, grouped by subagent (mocked backends)
    subagents.py          # subagent definitions + delegate_to_* tool wrappers
    supervisor.py          # the supervisor agent
frontend/
  index.html             # browser UI ("Dispatch" console) — vanilla HTML/CSS/JS, no build step
tests/
  test_support_endpoint.py  # endpoint tests with the LLM call mocked
```

## Browser UI ("Dispatch" console)

`frontend/index.html` is a small, dependency-free chat UI that talks to
`/support` directly from the browser. It's served automatically by the
FastAPI app itself (mounted at `/`), so once `uvicorn` is running, open:

```
http://localhost:8000/
```

Set your API key under the "connection" panel (top right) — it's kept
in memory only, never written to localStorage, so it clears on refresh.
Send a message and watch the routing panel on the right: all three
specialists pulse amber while the supervisor is deciding, then whichever
one actually handled the request lights up green, sourced straight from
the `subagents_used` field the API returns. This is a direct visualization
of the supervisor → subagent architecture from the tutorial, not just a
generic chat window.

The UI can also be pointed at a differently-hosted API by changing "API
base URL" in the connection panel — CORS is enabled on the backend
(`allow_origins=["*"]`) to support that during development; lock this down
to your real frontend origin before deploying either piece to production.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
uv sync
cp .env.example .env
# edit .env: set GOOGLE_API_KEY and API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

## Try it

Easiest: open `http://localhost:8000/` in a browser (see "Browser UI" above).

Or via curl:

```bash
curl -X POST http://localhost:8000/support \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your API_KEY from .env>" \
  -d '{
    "customer_id": "cust_001",
    "message": "I was charged twice, can you refund invoice INV-9001 for $49?"
  }'
```

Continue the same conversation by re-sending the `session_id` you get back
in the response — the supervisor's checkpointer remembers prior turns.

## Exercise for next time

The supervisor currently delegates one subagent call at a time (the LLM
decides to call one tool, waits, maybe calls another). Try modifying the
supervisor prompt and tool wiring so that when a request clearly spans
two domains (e.g. "refund my order AND tell me why it hasn't shipped"),
both delegations fire concurrently via `asyncio.gather` instead of
sequentially. This is the "router" pattern.
