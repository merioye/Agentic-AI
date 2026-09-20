# Expense Approval Copilot — LangGraph Mastery Project

A single LangGraph pipeline that demonstrates dynamic map-reduce (`Send`),
human-in-the-loop pausing (`interrupt` / `Command(resume=...)`), thread-scoped
persistence (checkpointer), and cross-thread long-term memory (`Store`) — all
exposed through FastAPI with SSE streaming and a browser UI.

## Architecture

```
START -> intake -> [Send: one review_item per line item, in parallel]
       -> review_item (N parallel) -> aggregate
       -> route_after_aggregate -> human_approval   OR   auto_approve
       -> finalize -> END
```

- `app/graph.py` — the whole pipeline: state schema, nodes, `Send` fan-out,
  `interrupt()`-based approval gate, `Store`-backed long-term memory in `finalize`
- `app/main.py` — FastAPI: `/api/expense/submit`, `/api/expense/resume` (both SSE),
  `/api/expense/history/{employee}` (reads the Store), `/api/expense/{thread_id}/timeline`
  (time-travel debug view over checkpoint history)
- `static/index.html` — browser UI: submission form, live progress log, an
  approval card that appears when the graph pauses, and history/timeline panels

## Setup

```bash
cd 6_langgraph
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv sync

cp .env.example .env
# set GOOGLE_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000**. Try:

1. **Auto-approve path**: employee "alex", one line item — category "software",
   $40, "Editor license", receipt checked. Should auto-approve (no flags, under
   the $500 threshold).
2. **Human-in-the-loop path**: employee "sam", a line item in "entertainment"
   for $200 with no receipt. Watch the log stream each item being reviewed,
   then the approval card appears — approve or reject it, add a note, and
   watch the stream resume and finalize from wherever it paused (potentially
   a completely separate HTTP request from the one that started it, which is
   the whole point of checkpointer-backed `interrupt()`).
3. Click **"View checkpoint timeline"** on a finished report to see every
   superstep LangGraph recorded for that thread — this is `get_state_history`,
   the time-travel debugging primitive.
4. Load **"Approval history"** for an employee you've submitted reports for —
   this reads back from the `Store`, which persists across threads (unlike the
   checkpointer, which is scoped to one thread/run).

## What this demonstrates

- **`Send`** for dynamic map-reduce: the number of parallel `review_item` runs
  is decided at runtime from however many line items were submitted, not fixed
  at compile time.
- **Reducer correctness**: `flagged_items: Annotated[list[...], operator.add]`
  is what lets N parallel branches all append to the same field without
  clobbering each other.
- **`interrupt()` / `Command(resume=...)`** for a real approval gate, including
  resuming from a _different_ HTTP request than the one that paused it — the
  checkpointer is what makes that possible.
- **Idempotency around `interrupt()`**: everything in `human_approval` before
  the `interrupt()` call is a pure recomputation from state, safe to re-run
  when the node restarts from the top on resume.
- **`Store`** for long-term, cross-thread memory (`finalize` writes; the
  history endpoint reads), independent of any single thread's checkpoints.
- **`get_state_history`** for time-travel debugging over a finished run.
- Three-way SSE (`custom` progress events + `updates` node/interrupt signals)
  consumed by a plain `fetch` + `ReadableStream` client.

## Known limitations (by design, for a tutorial project)

- `InMemorySaver` / `InMemoryStore` lose all data on server restart — swap for
  `PostgresSaver` / `PostgresStore` in a real deployment.
- No auth on the API, and no TTL sweep for abandoned interrupted threads —
  both are called out as must-haves before any real deployment.
- The policy classification is a single LLM call per line item with no retry
  policy configured — add a `RetryPolicy` on `review_item` before production use.
