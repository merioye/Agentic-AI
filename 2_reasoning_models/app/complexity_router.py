"""
Dynamic effort routing. Two tiers, cheapest first:
  1. Heuristic - free, instant, catches the obvious cases.
  2. Cheap-model classification - one fast_model call, only when the
     heuristic isn't confident.
Never default to "high" or above without one of these actually saying so
- that's the whole point: earn the extra cost per-question, don't assume it.
"""
from __future__ import annotations

import logging
from typing import Literal, cast

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger("reasoning.router")

EffortLevel = Literal["minimal", "low", "medium", "high"]

class ComplexityClassification(BaseModel):
    """A cheap model's judgment of how much reasoning effort a question warrants."""

    effort: EffortLevel = Field(
        description=(
            "'low' for straightforward factual/conversational questions; "
            "'medium' for multi-step analysis, non-trivial math, or several "
            "interacting constraints; 'high' only for genuinely hard problems "
            "where correctness matters than speed/cost."
        )
    )
    reason: str = Field(description="One short phrase for why")


def heuristic_classify(question: str) -> EffortLevel | None:
    """Returns a confident classification, or None to fall through to the
    LLM classifier. See config.py's high_effort_keywords/length threshold
    for the tunable knobs here."""
    settings = get_settings()
    q_lower = question.lower()

    if any(kw in q_lower for kw in settings.high_effort_keywords):
        return "high"

    has_digits = any(c.isdigit() for c in question)
    if len(question) < 40 and not has_digits:
        return "minimal" # short, no numbers - near certainly simple

    if len(question) > settings.heuristic_length_threshold:
        return None # long enough to be genuinely ambiguous - ask the classifier

    return None

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        model = ChatGoogleGenerativeAI(
            model=get_settings().fast_model,
            google_api_key=get_settings().google_api_key,
        )
        _classifier = model.with_structured_output(ComplexityClassification)
    return _classifier


async def llm_classify(question: str) -> EffortLevel:
    classifier = _get_classifier()
    result = await classifier.ainvoke(
        f"Classify the reasoning effort for this question warrants:\n\n{question}"
    )
    classification = cast(ComplexityClassification, result)
    logger.info("LLM-classified effort=%s reason=%r for question=%r", classification.effort, classification.reason, question[:80])
    return classification.effort


async def classify_effort(question: str) -> EffortLevel:
    settings = get_settings()
    heuristic_result = heuristic_classify(question)
    if heuristic_classify is not None:
        logger.info("Heuristic-classified effort=%s for question=%r", heuristic_result, question[:80])

    if settings.use_llm_classifier_fallback:
        return await llm_classify(question)

    return settings.default_effort # type: ignore[return-type]