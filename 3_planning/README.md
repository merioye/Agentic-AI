# Planning & Goal Decomposition — Day 3 Reference Application

A complete, runnable implementation of both major agentic planning
patterns, sharing the same tools and the same goal so they're directly
comparable.

```
3_planning/
├── app/
│   ├── main.py                    FastAPI: /todo/chat, /plan-execute/plan, /plan-execute/run
│   ├── config.py                   Settings
│   ├── tools.py                     Shared tools + reference dataset (extends Day 2's)
│   ├── todo_agent.py                 Pattern 1: create_agent + TodoListMiddleware
│   ├── plan_execute.py                Pattern 2: planner/executor/replanner, parallel execution
│   └── schemas.py                      Request/response models
├── ui/index.html                 Side-by-side console: run both patterns on one goal, plan approval flow
├── cli.py                          CLI: todo / plan / run / compare
├── evals/                            Decomposition coverage + efficiency, both patterns, side by side
└── .env.example
```

## The one thing to understand before reading the code

**These are two different architectures, not two configurations of one
thing.** `app/todo_agent.py` is a normal `create_agent` with one extra
middleware (`TodoListMiddleware`) — the model plans and re-plans
in-context, one continuous loop. `app/plan_execute.py` is a genuinely
separate orchestrator: a planner call produces a structured, dependency-
aware plan _before any execution happens_, an executor runs every
currently-unblocked step **concurrently**, and a replanner decides after
each round whether the plan needs revising or the goal is done. Guide §2
is the full tradeoff table; the CLI's `compare` command runs the identical
goal through both so you can see it rather than take it on faith.

## 1. Setup

```bash
cd 3_planning
python -m venv .venv && source .venv/bin/activate
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

**Compare both patterns on the same goal, from the CLI:**

```bash
python cli.py compare
```

(Uses a built-in default QBR goal — pass your own as an argument to
override it.) Watch the plan-and-execute side actually dispatch the four
department-related steps concurrently (guide §4) vs. the todo-list agent
working through its list one item at a time.

**See a plan before anything executes (the human-approval flow, guide §5):**

```bash
python cli.py plan
# review the printed step table, then:
python cli.py run
```

**Browser UI:** http://localhost:8000/ui/ — "Run both" fires the goal at
both patterns at once, rendered side by side. "Preview plan only" shows
the plan-and-execute side's plan without executing anything, with an
"Approve & execute this plan" button that becomes the actual approved-plan
execution — the same request the CLI's two-call `plan` → `run` flow makes.

## 4. Run the comparison eval

```bash
python -m evals.run_evals
```

Runs the QBR goal through both patterns and reports **decomposition
coverage** (did the plan/todos actually reference all four required
departments?) and **efficiency** (step/todo counts, parallel-ready step
count, replan cycles, elapsed time) side by side — guide §6, made
concrete and runnable rather than left as an abstract checklist.

---

## Production notes

- **The plan-execute orchestrator is a hand-rolled async loop, not a
  compiled `StateGraph`** — deliberately, to support two entry points
  (fresh goal vs. pre-approved plan) without fighting a compiled graph's
  fixed entry point. See the docstring at the top of `app/plan_execute.py`
  for the full reasoning and what you'd gain from swapping to a real
  `StateGraph` (LangGraph-native tracing, checkpointed `interrupt()`-based
  approval instead of this project's two-call pattern) if you need those
  specifically.
- **No checkpointer, no multi-turn memory, no auth** — same scoping
  decisions as Day 5, for the same reason: this is a planning-pattern
  demonstration/benchmarking tool, not a long-running conversational
  product. Add a checkpointer the same way Day 1 does if you need to
  persist plan state across process restarts or support pausing a
  long-running plan for hours/days between the plan and run calls.
- **The executor's per-step sub-agent uses a cheaper model
  (`EXECUTOR_MODEL`) than the planner/replanner** — the concrete version
  of guide §2.2's "cheaper execution" claim. If your steps need more
  reasoning power than a fast model provides, raise it, but the general
  principle (spend your best model on planning/replanning, not on every
  individual step) is worth keeping even when you do.
- **`max_replan_cycles` and `max_plan_steps` are real cost/runaway-loop
  guardrails**, same spirit as `ModelCallLimitMiddleware` elsewhere in
  this series — a planner or replanner that keeps finding "just one more
  thing" to add is a real, not hypothetical, failure mode once you're
  running this against genuinely open-ended goals rather than the
  bounded QBR scenario here.
