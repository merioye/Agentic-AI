"""
Shared tools - deliberately the same tools + reference dataset as Day 2
(quantitative analysis), extended to four departments specifically so the
QBR goal has real decomposable structure: three independent data-gathering
steps plus a synthesis step is exactly the shape that makes explicit
planning(vs. one-step-at-a-time reaction) pay off.
"""
from __future__ import annotations

import ast
import operator

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

# ---------------------------------------------------------------------------
# safe arithmetic evaluator - same security descipline as Day 2: as small
# AST allowlist, never Python's eval(), since the "expression" is 
# LLM-generated untrusted input.
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression element: {ast.dump(node)}")


@tool
def evaluate_expression(expression: str) -> str:
    """Evaluate a numeric arithmetic (+, -, *, /, **, %, parenthesis
    only). Use this for any calculation rather than doing arithmetic yourself."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"
    return f"{expression} = {result}"


_REFERENCE_DATA = {
    "marketing": {"q3": 482_000, "q2": 430_000, "alert_threshold": 550_000, "unit": "USD"},
    "sales": {"q3": 1_240_000, "q2": 1_150_000, "alert_threshold": 1_500_000, "unit": "USD"},
    "engineering_cloud_spend": {"q3": 96_000, "q2": 88_000, "alert_threshold": 120_000, "unit": "USD"},
    "support_ticket_volume": {"q3": 8_400, "q2": 7_900, "alert_threshold": 10_000, "unit": "tickets"},
}


@tool
def lookup_reference_data(key: str) -> str:
    """Look up a known department's Q2/Q3 metric and its alert threshold.
    Valid keys: marketing, sales, engineering_cloud_spend, support_ticket_volume.
    Always call this before reasoning about any of these metrics."""
    entry = _REFERENCE_DATA.get(key.strip().lower().replace(" ", "_"))
    if not entry:
        return f"No reference data found for '{key}'. Valid keys: {list(_REFERENCE_DATA.keys())}"
    return (
        f"{key}: Q2 = {entry['q2']:,} {entry['unit']}, Q3 = {entry['q3']:,} {entry['unit']}, "
        f"alert threshold = {entry['alert_threshold']:,} {entry['unit']}"
    )


# ---------------------------------------------------------------------------
# Report-writing tool - used only by the TodoListMiddleware agent (the
# plan-and-execute graph accumulates step results directly in its own
# state instead). Appends via the operator.add reducer on QBRAgentState,
# same accumulation pattern as prior days' citations/notes fields.
# ---------------------------------------------------------------------------
@tool
def write_report_section(title: str, content: str, runtime: ToolRuntime) -> Command:
    """Record a completed section of the QBR report. Call this once per
    department (or once for a synthesis/recommendations section) as you
    finish reasoning about it - don't wait until the very end to write
    everything at once, so progress is visible if the task is interrupted."""
    section = {"title": title, "content": content}
    return Command(
        update={
            "report_sections": [section],
            "messages": [
                {"role": "tool", "content": f"Recorded section: {title}", "tool_call_id": runtime.tool_call_id}
            ]
        }
    )


ALL_TOOLS = [lookup_reference_data, evaluate_expression, write_report_section]
EXECUTOR_TOOLS = [lookup_reference_data, evaluate_expression] # plan-execute's per-step executor doesn't need write_report_section