#!/usr/bin/env python3
"""
CLI for the reasoning-effort router.

    python cli.py chat
    python cli.py chat --force-effort high
    python cli.py compare "your question here"   # runs it at every effort level back to back
"""
import argparse
import json

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()

EFFORT_COLORS = {"minimal": "green", "low": "yellow", "medium": "red", "high": "bold red"}


def cmd_chat(args) -> None:
    client = httpx.Client(base_url=args.base_url, timeout=120.0)
    console.print(
        Panel.fit(
            "[bold]Reasoning Effort Router[/bold] (streaming)\n"
            + ("Effort: forced to " + args.force_effort if args.force_effort else "Effort: auto-routed per question")
            + "\nType 'exit' to quit.",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("[bold green]you[/bold green]")
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        body = {"message": user_input}
        if args.force_effort:
            body["force_effort"] = args.force_effort

        effort_used = None
        console.print("[bold magenta]agent[/bold magenta]: ", end="")
        try:
            with client.stream("POST", "/chat/stream", json=body) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])
                    if event["type"] == "effort_selected":
                        effort_used = event["effort"]
                    elif event["type"] == "token":
                        console.print(event["text"], end="")
        except httpx.HTTPError as e:
            console.print(f"[red]Error: {e}[/red]")
            continue
        console.print()

        color = "white"
        if effort_used is not None:
            color = EFFORT_COLORS.get(effort_used, "white")
        console.print(f"[dim]effort: [{color}]{effort_used}[/{color}][/dim]")


def cmd_compare(args) -> None:
    """Runs the same question at every effort level, side by side — the
    concrete way to SEE what effort buys you, rather than trusting it does
    something on faith (guide §7)."""
    client = httpx.Client(base_url=args.base_url, timeout=180.0)
    table = Table(title=f"Effort comparison: {args.question!r}")
    table.add_column("Effort")
    table.add_column("Thinking tokens")
    table.add_column("Answer (truncated)")

    for effort in ["low", "medium", "high"]:
        resp = client.post("/chat", json={"message": args.question, "force_effort": effort})
        if resp.status_code != 200:
            table.add_row(effort, "-", f"[red]error: {resp.text[:80]}[/red]")
            continue
        data = resp.json()
        table.add_row(
            effort,
            str(data.get("thinking_tokens", "-")),
            (data["message"][:100] + "…") if len(data["message"]) > 100 else data["message"],
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reasoning Effort Router — CLI")
    parser.add_argument("--base-url", default="http://localhost:8000")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--force-effort", default=None, choices=["standard", "high", "xhigh", "max", "none"])
    p_chat.set_defaults(func=cmd_chat)

    p_compare = sub.add_parser("compare", help="Run one question at every effort level")
    p_compare.add_argument("question")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
