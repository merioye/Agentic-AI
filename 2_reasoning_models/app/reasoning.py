"""
Two responsibilities:
  1. model_for_effort() - build a model instance bound to a given dynamic
     thinking effort level. 'none" disables thinking.
  2. extract_reasoning() - pull the reasoning summary + billed thinking-token
     count out of a response, for the observability surfaced in app/main.py
     and the UI

SCOPING NOTE: extract_reasoning() reports figures from the LAST model call
in a turn only. A turn with multiple tool-calling round-trips spends
additional (billed) reasoning tokens in earlier steps that are'nt summed
here - this keeps the implementation simple and honest rather than
attempting fragile cross-step aggregation.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings

@lru_cache(maxsize=1)
def _base_model(thinking_level: str):
    return ChatGoogleGenerativeAI(
        model=get_settings().primary_model,
        google_api_key=get_settings().google_api_key,
        thinking_level=thinking_level
    )

def model_for_effort(thinking_level: str) -> ChatGoogleGenerativeAI:
    return _base_model(thinking_level)


def extract_reasoning(messages: list) -> tuple[str | None, int]:
    """Walks backward to the last AI message and pulls out any
    thinking/redacted_thinking content plus the billed thinking-token
    count from its usage metadata. Returns (summary_text_or_None, tokens)"""
    last_ai_message = None
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            last_ai_message = msg
            break
    if last_ai_message is None:
        return None, 0

    summary_parts = []
    for block in getattr(last_ai_message, "content_blocks", None) or []:
        btype = block.get("type")
        if btype in ("reasoning", "thinking"):
            text = block.get("reasoning") or block.get("thinking") or ""
            if text:
                summary_parts.append(text)
        elif btype == "redacted_thinking" or block.get("redacted"):
            summary_parts.append("[reasoning redacted by safety system]")

    usage = getattr(last_ai_message, "usage_metadata", None) or {}
    output_details = usage.get("output_token_details") or usage.get("output_tokens_details") or {}
    thinking_tokens = output_details.get("thinking_tokens") or output_details.get("reasoning_tokens") or output_details.get("reasoning") or 0

    return ("\n\n".join(summary_parts) or None, thinking_tokens)