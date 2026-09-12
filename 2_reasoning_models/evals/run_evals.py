#!/usr/bin/env python3
"""
Runs every golden case at every effort level and reports accuracy,
thinking-token cost, and latency side by side — needs a real
GOOGLE_API_KEY, and costs real money (this is the whole point: an
empirical answer to "is high effort worth it here," not a guess).

    python -m evals.run_evals
"""
import asyncio
import sys
import time

from app.config import get_settings
from app.reasoning_agent import AgentContext, build_agent, run_turn
from evals.golden_cases import CASES

EFFORT_LEVELS_TO_COMPARE = ["low", "medium", "high"]


async def run_one(agent, case, effort: str) -> dict:
    context = AgentContext(user_id=None, conversation_id=f"eval-{case.name}-{effort}", effort=effort)
    start = time.perf_counter()
    try:
        result = await run_turn(agent, message=case.question, context=context)
    except Exception as exc:
        return {"effort": effort, "correct": False, "error": str(exc), "thinking_tokens": None, "latency_s": None}
    latency = time.perf_counter() - start

    answer_lower = result["message"].lower()
    correct = any(s.lower() in answer_lower for s in case.expected_answer_substrings)

    return {
        "effort": effort, "correct": correct, "error": None,
        "thinking_tokens": result["thinking_tokens"], "latency_s": round(latency, 2),
        "answer": result["message"],
    }


async def main() -> int:
    settings = get_settings()
    if not settings.google_api_key:
        print("No GOOGLE_API_KEY set — evals need a real model. Aborting.")
        return 1

    agent = await build_agent()

    print(f"Running {len(CASES)} case(s) x {len(EFFORT_LEVELS_TO_COMPARE)} effort level(s)...\n")

    overall_rows = []
    for case in CASES:
        print(f"=== {case.name} ===")
        print(f"    {case.question}")
        if case.note:
            print(f"    ({case.note})")
        for effort in EFFORT_LEVELS_TO_COMPARE:
            row = await run_one(agent, case, effort)
            overall_rows.append({**row, "case": case.name})
            status = "PASS" if row["correct"] else "FAIL"
            if row["error"]:
                print(f"    [{effort:>8}] ERROR: {row['error']}")
            else:
                print(
                    f"    [{effort:>8}] {status}  "
                    f"thinking_tokens={row['thinking_tokens']!s:>6}  "
                    f"latency={row['latency_s']}s"
                )
                if not row["correct"]:
                    print(f"               answer: {row['answer'][:150]}")
        print()

    # Summary: accuracy per effort level across all cases, so the
    # "is high effort worth it, in general, for THIS problem set" question
    # has a direct answer rather than requiring you to eyeball the table above.
    print("=== Summary: accuracy by effort level ===")
    for effort in EFFORT_LEVELS_TO_COMPARE:
        rows = [r for r in overall_rows if r["effort"] == effort and r["error"] is None]
        if not rows:
            continue
        correct = sum(r["correct"] for r in rows)
        avg_tokens = sum(r["thinking_tokens"] or 0 for r in rows) / len(rows)
        avg_latency = sum(r["latency_s"] or 0 for r in rows) / len(rows)
        print(
            f"  {effort:>8}: {correct}/{len(rows)} correct, "
            f"avg thinking_tokens={avg_tokens:.0f}, avg latency={avg_latency:.2f}s"
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
