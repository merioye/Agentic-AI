"""
Two tools, deliberately paired so a realistic question forces the model to
look something up, reason about what it got back, THEN decide what to
calculate - not execute a fixed lookup-then-calculate sequence it could
have planned before ever seeing the data. This is what makes interleaved
thinking visible rather theoretical.
"""
from __future__ import annotations

import ast
import operator

from langchain.tools import tool

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator - NOT Python's eval(). Then agent's calculator
# input is LLM-generated text; restricting to a small AST node allowlist
# (numbers + basic operators only, no names/calls/attributes) means there
# no code-injection surface even though the "expression" is untrusted input.
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
    """Evaluate a numeric arithmetic expression (+, -, *, /, **, %, parentheses 
    only - no variables or function calls). Use this for any calculation
    rather than doing arithmetic yourself, to guarantee accuracy."""
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"
    return f"{expression} = {result}"


# ---------------------------------------------------------------------------
# Reference data - deliberately requires a lookup BEFORE the calculation
# that depends on it can happen, and the department names/thresholds
# aren't things the models could plausibly know or guess correctly.
# ---------------------------------------------------------------------------
_REFERENCE_DATA = {
    "marketing": {"q3_revenue": 482_000, "alert_threshold": 550_000, "unit": "USD"},
    "sales": {"q3_revenue": 1_240_000, "alert_threshold": 1_500_000, "unit": "USD"},
    "engineering_cloud_spend": {"q3_revenue": 96_000, "alert_threshold": 120_000, "unit": "USD"},
    "support_ticket_volume": {"q3_revenue": 8_400, "alert_threshold": 10_000, "unit": "tickets"},
}


@tool
def lookup_reference_data(key: str) -> str:
    """Look up a known quarterly metric and its alert threshold. Valid keys:
    marketing, sales, engineering_cloud_spend, support_ticket_volume.
    Always call this before reasoning about any of these metrics - never
    guess or assume a figure."""
    entry = _REFERENCE_DATA.get(key.strip().lower().replace(" ", "_"))
    if not entry:
        return f"No reference data found for '{key}'. Valid keys: {list(_REFERENCE_DATA.keys())}"
    return (
        f"{key}: Q3 value = {entry['q3_revenue']:,} {entry['unit']}, "
        f"alert threshold = {entry['alert_threshold']:,} {entry['unit']}"
    )


ALL_TOOLS = [evaluate_expression, lookup_reference_data]