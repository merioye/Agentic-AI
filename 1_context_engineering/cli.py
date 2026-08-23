#!/usr/bin/env python3
"""
Interactive CLI client for the support agent API.

Streams tokens from /chat/stream as they're generated instead
of waiting for the full response — you should see the agent's reply appear
incrementally rather than all at once.

Usage:
    python cli.py --api-key customer-key-1
"""
import argparse
import json
import uuid

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def stream_chat(client: httpx.Client, conversation_id: str, message: str) -> dict:
    """POST to /chat/stream, print tokens as they arrive, return the final event."""
    final_event = None
    console.print("[bold magenta]agent[/bold magenta]: ", end="")
    with client.stream(
        "POST", "/chat/stream", json={"conversation_id": conversation_id, "message": message}
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            if event["type"] == "token":
                console.print(event["text"], end="")
            elif event["type"] == "done":
                final_event = event
    console.print()  # newline after the streamed reply
    return final_event or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Acme Support Agent — CLI client")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key",
        default="guest-key",
        help="guest-key | customer-key-1 | customer-key-2 | admin-key",
    )
    parser.add_argument("--conversation-id", default=None)
    args = parser.parse_args()

    conversation_id = args.conversation_id or f"cli-{uuid.uuid4().hex[:8]}"
    headers = {"X-API-Key": args.api_key}

    console.print(
        Panel.fit(
            f"[bold]Acme Support Agent[/bold] (streaming)\n"
            f"conversation_id: [cyan]{conversation_id}[/cyan]\n"
            f"identity: [cyan]{args.api_key}[/cyan]\n\n"
            f"Type your message, or 'exit' to quit.",
            border_style="blue",
        )
    )

    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=60.0)

    while True:
        try:
            user_input = Prompt.ask("[bold green]you[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("[dim]bye[/dim]")
            break

        try:
            data = stream_chat(client, conversation_id, user_input)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Error {e.response.status_code}: {e.response.text}[/red]")
            continue
        except httpx.RequestError as e:
            console.print(f"[red]Connection error: {e}[/red]")
            continue

        if data.get("requires_approval"):
            console.print(
                Panel(
                    f"[yellow]Approval needed:[/yellow] {data.get('pending_action')}",
                    border_style="yellow",
                )
            )
            decision = Prompt.ask("Approve this action?", choices=["approve", "reject"])
            resp = client.post(
                "/chat/resume",
                json={"conversation_id": conversation_id, "decision": decision},
            )
            resp.raise_for_status()
            result = resp.json()
            console.print("[bold magenta]agent[/bold magenta]:", result["message"])
            data = result

        if data.get("escalate_to_human"):
            console.print("[yellow]⚠ This conversation was flagged for human follow-up.[/yellow]")


if __name__ == "__main__":
    main()
