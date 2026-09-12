# Reasoning Effort Router — Day 2 Reference Application

A complete, runnable implementation of production reasoning-model usage:
FastAPI + LangChain v1.x agent with **dynamic-thinking effort
routing**, a tool-use scenario built to make interleaved thinking visible,
and a benchmark comparing accuracy/cost/latency across effort levels.

```
day05-reasoning-models/
├── app/
│   ├── main.py                    FastAPI: classifies effort BEFORE invoking the agent
│   ├── config.py                   Settings + effort-routing thresholds
│   ├── complexity_router.py         Heuristic-first, cheap-LLM-fallback classifier
│   ├── reasoning.py                  Effort->model factory + thinking-block/token extraction
│   ├── reasoning_agent.py             create_agent: effort applied via wrap_model_call
│   ├── tools.py                        Calculator + reference-data lookup (designed to force interleaving)
│   └── schemas.py                       Request/response models
├── ui/index.html                 Effort badge + thinking-token count + reasoning-summary panel
├── cli.py                          CLI: chat, or `compare` — one question at every effort level
├── evals/                            Accuracy/cost/latency benchmark across effort levels (needs a real LLM)
└── .env.example
```

## The one thing to understand before reading the code

**Effort is chosen once, before the agent runs, and travels in as Runtime
Context — not inside a middleware hook that itself calls an LLM.**
`app/main.py` classifies the incoming question (`app/complexity_router.py`)
and puts the result on `AgentContext.effort`; `apply_reasoning_effort` in
`app/reasoning_agent.py` is a plain **asynchronous** `wrap_model_call` that
just reads `request.runtime.context.effort` and swaps in the right model.
This keeps the middleware itself simple and consistent with every prior
day's middleware shape, and treats "how hard should we think about this"
as what it actually is: a per-request decision made once, not a
graph-internal concern.

## 1. Setup

```bash
cd 02_reasoning_models
uv sync

cp .env.example .env
# edit .env: GOOGLE_API_KEY
```

## 2. Run the API

```bash
uvicorn app.main:app --reload
```

- API: http://localhost:8000 · docs: http://localhost:8000/docs
- Browser UI: http://localhost:8000/ui/

## 3. Try it

**See the router in action:**

```bash
python cli.py chat
you: hi
# -> effort: minimal (heuristic: short, no keywords, no digits)

you: If Sales grows 10% per quarter for the next two quarters starting
     from its Q3 figure, will it cross its alert threshold, and by how much?
# -> effort: high (heuristic keyword match: none of the trigger words,
#    so this one actually goes through the LLM classifier fallback —
#    watch the server log to see "LLM-classified effort=...")
```

**See what effort actually buys you, empirically, on one question:**

```bash
python cli.py compare "If Sales grows 10% per quarter for the next two \
quarters starting from its Q3 figure, will it cross its alert threshold, \
and by how much?"
```

Prints a table: effort level, thinking tokens billed, and the answer —
side by side, so "does high effort actually help here" has a real answer
instead of a guess.

**Force a specific effort level** (bypass the router entirely, useful for
testing/demos):

```bash
python cli.py chat --force-effort high
```

**Browser UI:** http://localhost:8000/ui/ — shows the effort badge, billed
thinking-token count, and a collapsible reasoning-summary panel (with the
transparency caveat from guide §3 shown inline, not hidden in a tooltip).

## 4. Run the effort-comparison benchmark

```bash
python -m evals.run_evals
```

Runs 4 golden problems (from trivial to genuinely multi-step) at
`low`, `medium`, and `high`, reporting **accuracy, thinking tokens,
and latency side by side** — then a summary table of accuracy-by-effort
across the whole set. This is the concrete tool for tuning
`app/config.py`'s routing thresholds against real evidence rather than
intuition.

---

## Production notes

- **Streaming vs. detail tradeoff, stated explicitly.** `/chat/stream`
  reports which effort level was chosen (`effort_selected` event) but NOT
  `thinking_tokens`/`reasoning_summary` — there's no checkpointer in this
  project to re-fetch final state from after a stream completes (see
  `reasoning_agent.stream_turn`'s docstring). Use `/chat` (non-streaming)
  when you need those numbers. A production system would pick one pattern
  consistently per use case rather than needing both, the way this
  teaching app's UI does (openly, in a code comment, not silently).
- **This project has no multi-turn conversation memory** — no
  checkpointer, unlike Days 1/1.5/3/3.5/4. Each request is evaluated
  independently. That's a deliberate scope decision (a quantitative
  analysis/benchmarking tool, not an extended support dialogue) — if you
  need both effort routing AND conversation memory, add a checkpointer
  the same way Day 1 does; nothing about the effort-routing mechanism
  changes.
- **No auth in this project** — see the comment at the top of
  `app/main.py`. There's no access-control-relevant data here, unlike the
  days that handle orders/documents/media.
- **The routing thresholds in `config.py` are starting points, not
  tuned values.** `high_effort_keywords`, `heuristic_length_threshold` —
  tune these against your own traffic and the eval benchmark's accuracy
  numbers, the same way you'd tune any classifier threshold. Shipping the
  defaults unchanged and assuming they're calibrated for your question
  distribution is the single most likely way this pattern underperforms
  in practice.
- **`model_for_effort`'s `.bind()` call is the one place to check first**
  if you're on a different langchain-anthropic version than this was
  written against (Aug 2026) — see its docstring in `app/reasoning.py`.
  Everything else in the app is agnostic to how that one function is
  implemented.
