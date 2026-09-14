#!/usr/bin/env python3
"""
CLI for comparing the two planning patterns.

    python cli.py todo "Produce a QBR covering all four departments..."
    python cli.py plan "Produce a QBR covering all four departments..."   # plan only, no execution
    python cli.py run "..."                                                 # plan + execute fresh
    python cli.py compare "..."                                              # both patterns, side by side, with timing
"""
import argparse
import time

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()

DEFAULT_GOAL = (
    "Produce a QBR covering Marketing, Sales, Engineering Cloud Spend, and "
    "Support Ticket Volume. For each, note the Q2->Q3 percentage change and "
    "whether Q3 crosses the alert threshold. Recommend next steps for any "
    "department at or near its threshold."
)


def cmd_todo(args) -> None:
    client = httpx.Client(base_url=args.base_url, timeout=180.0)
    console.print("[dim]Running via TodoListMiddleware agent...[/dim]")
    start = time.perf_counter()
    resp = client.post("/todo/chat", json={"goal": args.goal})
    elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        console.print(f"[red]Failed: {resp.text}[/red]")
        return
    data = resp.json()

    console.print(Panel(data["message"], title="Final message", border_style="green"))

    if data["todos"]:
        tree = Tree(f"Todos ({data['todos_completed']}/{data['todos_total']} completed)")
        for t in data["todos"]:
            icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(t["status"], "•")
            tree.add(f"{icon} {t['content']}")
        console.print(tree)

    if data["report_sections"]:
        console.print("\n[bold]Report sections:[/bold]")
        for s in data["report_sections"]:
            console.print(f"  [cyan]{s['title']}[/cyan]: {s['content']}")

    console.print(f"\n[dim]elapsed: {elapsed:.1f}s[/dim]")


def cmd_plan(args) -> None:
    client = httpx.Client(base_url=args.base_url, timeout=60.0)
    resp = client.post("/plan-execute/plan", json={"goal": args.goal})
    if resp.status_code != 200:
        console.print(f"[red]Failed: {resp.text}[/red]")
        return
    data = resp.json()
    _print_plan(data["plan"])
    console.print("\n[dim]Plan only — nothing executed. Use 'run' to execute.[/dim]")


def cmd_run(args) -> None:
    client = httpx.Client(base_url=args.base_url, timeout=300.0)
    console.print("[dim]Running via Plan-and-Execute graph...[/dim]")
    start = time.perf_counter()
    resp = client.post("/plan-execute/run", json={"goal": args.goal})
    elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        console.print(f"[red]Failed: {resp.text}[/red]")
        return
    data = resp.json()

    console.print(Panel(data["final_message"] or "(no final message)", title="Final message", border_style="green"))
    _print_plan(data["plan"])
    console.print(f"\n[dim]replan cycles: {data['replan_cycles']}, elapsed: {elapsed:.1f}s[/dim]")


def _print_plan(steps) -> None:
    table = Table(title="Plan")
    table.add_column("id")
    table.add_column("description")
    table.add_column("depends_on")
    table.add_column("status")
    for s in steps:
        status_color = {"completed": "green", "failed": "red", "pending": "yellow", "in_progress": "cyan"}.get(s["status"], "white")
        table.add_row(s["id"], s["description"], ", ".join(s["depends_on"]) or "-", f"[{status_color}]{s['status']}[/{status_color}]")
    console.print(table)


def cmd_compare(args) -> None:
    """Runs the SAME goal through both patterns, back to back, timed —
    the concrete way to see guide §2.3's tradeoff table hold up (or not)
    on a real run, rather than trusting it on faith."""
    goal = args.goal or DEFAULT_GOAL
    console.print(Panel.fit(f"[bold]Comparing planning patterns[/bold]\n{goal}", border_style="blue"))

    console.print("\n[bold cyan]== Todo-list pattern ==[/bold cyan]")
    cmd_todo(argparse.Namespace(base_url=args.base_url, goal=goal))

    console.print("\n[bold magenta]== Plan-and-Execute pattern ==[/bold magenta]")
    cmd_run(argparse.Namespace(base_url=args.base_url, goal=goal))


def main() -> None:
    parser = argparse.ArgumentParser(description="Planning & Goal Decomposition — CLI")
    parser.add_argument("--base-url", default="http://localhost:8000")
    sub = parser.add_subparsers(dest="command", required=True)

    p_todo = sub.add_parser("todo", help="Run via the TodoListMiddleware agent")
    p_todo.add_argument("goal", nargs="?", default=DEFAULT_GOAL)
    p_todo.set_defaults(func=cmd_todo)

    p_plan = sub.add_parser("plan", help="Plan only (no execution) — for review")
    p_plan.add_argument("goal", nargs="?", default=DEFAULT_GOAL)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="Plan and execute fresh")
    p_run.add_argument("goal", nargs="?", default=DEFAULT_GOAL)
    p_run.set_defaults(func=cmd_run)

    p_compare = sub.add_parser("compare", help="Run the same goal through both patterns")
    p_compare.add_argument("goal", nargs="?", default=None)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
