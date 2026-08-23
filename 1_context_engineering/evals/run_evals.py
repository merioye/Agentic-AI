#!/usr/bin/env python3
"""
Run the golden eval set against the real agent (real LLM calls — needs a
valid API key in .env). Use this after touching prompts, tool docstrings,
or middleware config, to catch regressions before they reach users.

    python -m evals.run_evals

For full observability into *why* a case failed (what the model actually
saw at each step), set these before running and inspect the trace in
LangSmith: https://smith.langchain.com

    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=...
    export LANGCHAIN_PROJECT=acme-support-agent-evals
"""
import asyncio
import sys
import uuid

from app.agent import AgentContext, build_agent, run_turn
from app.auth import _DEMO_USERS
from app.config import get_settings
from app.db import init_db
from evals.golden_cases import GOLDEN_CASES


async def run_case(agent, case) -> tuple[bool, list[dict], str | None]:
    user_id, role = _DEMO_USERS[case.api_key]
    context = AgentContext(user_id=user_id, role=role)
    thread_id = f"eval-{case.name}-{uuid.uuid4().hex[:6]}"

    responses = []
    try:
        for turn in case.turns:
            result = await run_turn(agent, thread_id=thread_id, message=turn, context=context)
            responses.append(result)
    except Exception as exc:  # noqa: BLE001 - want to report, not crash the whole run
        return False, responses, f"exception: {exc}"

    try:
        passed = bool(case.check(responses))
    except Exception as exc:  # noqa: BLE001
        return False, responses, f"check() raised: {exc}"

    return passed, responses, None


async def main() -> int:
    settings = get_settings()
    if not settings.google_api_key:
        print("No GOOGLE_API_KEY set — evals need a real model. Aborting.")
        return 1

    init_db()
    agent, checkpointer_cm = await build_agent()

    print(f"Running {len(GOLDEN_CASES)} golden case(s) against {settings.primary_model}...\n")

    failures = []
    for case in GOLDEN_CASES:
        passed, responses, error = await run_case(agent, case)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case.name}")
        if case.description:
            print(f"       {case.description}")
        if not passed:
            failures.append(case.name)
            if error:
                print(f"       error: {error}")
            else:
                print(f"       final response: {responses[-1] if responses else '(none)'}")
        print()

    await checkpointer_cm.__aexit__(None, None, None)

    total = len(GOLDEN_CASES)
    print(f"{total - len(failures)}/{total} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
