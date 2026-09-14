#!/usr/bin/env python3
"""
Runs the QBR goal through BOTH planning patterns and reports the guide §6
metrics side by side — needs a real GOOGLE_API_KEY, costs real money.

    python -m evals.run_evals

Checks:
  - Decomposition coverage: did the plan/todos actually reference all four
    departments the goal requires? (checkable independent of whether the
    final answer happened to be correct — a plan that skipped a
    department but got lucky elsewhere shouldn't score as "good planning.")
  - Efficiency: step/todo count, and for plan-execute specifically, how
    many steps ran in the first (parallel) executor batch vs. serially,
    and how many replan cycles were needed.
"""
import asyncio
import sys
import time

from app import plan_execute, todo_agent
from app.config import get_settings

GOAL = (
    "Produce a QBR covering Marketing, Sales, Engineering Cloud Spend, and "
    "Support Ticket Volume. For each, note the Q2->Q3 percentage change and "
    "whether Q3 crosses the alert threshold. Recommend next steps for any "
    "department at or near its threshold."
)

REQUIRED_DEPARTMENTS = ["marketing", "sales", "engineering", "support"]


def _coverage(texts: list[str]) -> dict[str, bool]:
    combined = " ".join(texts).lower()
    return {dept: dept in combined for dept in REQUIRED_DEPARTMENTS}


async def eval_todo_pattern() -> None:
    print("=== Todo-list pattern ===")
    agent = await todo_agent.build_agent()
    start = time.perf_counter()
    result = await todo_agent.run_goal(agent, goal=GOAL)
    elapsed = time.perf_counter() - start

    texts = [t["content"] for t in result["todos"]] + [s["content"] for s in result["report_sections"]]
    coverage = _coverage(texts)
    covered = sum(coverage.values())

    print(f"  todos: {result['todos_total']} total, {result['todos_completed']} completed")
    print(f"  report sections written: {len(result['report_sections'])}")
    print(f"  department coverage: {covered}/{len(REQUIRED_DEPARTMENTS)} {coverage}")
    print(f"  elapsed: {elapsed:.1f}s")
    print()


async def eval_plan_execute_pattern() -> None:
    print("=== Plan-and-Execute pattern ===")
    start = time.perf_counter()

    # Re-run planner_node alone first, purely to measure how many steps
    # were identified as immediately parallel-ready (guide §4) before any
    # execution has happened, rather than inferring it after the fact.
    plan_result = await plan_execute.planner_node({"goal": GOAL})
    initial_ready = plan_execute._get_ready_steps(plan_result["plan"])

    final_state = await plan_execute.run_plan(GOAL)
    elapsed = time.perf_counter() - start

    texts = [s["description"] for s in final_state["plan"]] + [s["content"] for s in final_state.get("report_sections", [])]
    coverage = _coverage(texts)
    covered = sum(coverage.values())

    print(f"  total steps: {len(final_state['plan'])}")
    print(f"  steps ready in parallel on the first pass: {len(initial_ready)}")
    print(f"  replan cycles: {final_state['replan_cycles']}")
    print(f"  department coverage: {covered}/{len(REQUIRED_DEPARTMENTS)} {coverage}")
    print(f"  elapsed: {elapsed:.1f}s")
    print()


async def main() -> int:
    settings = get_settings()
    if not settings.google_api_key:
        print("No GOOGLE_API_KEY set — evals need a real model. Aborting.")
        return 1

    await eval_todo_pattern()
    await eval_plan_execute_pattern()

    print("=== Interpretation ===")
    print(
        "Compare 'steps ready in parallel on the first pass' against the "
        "todo pattern's step-by-step nature (compare elapsed times above "
        "for whether that parallelism actually translated into a real "
        "wall-clock speedup on this run)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
