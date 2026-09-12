"""
Golden problems for the effort-vs-accuracy-vs-cost benchmark.
Every problem has a known correct numeric answer AND requires at least one
tool call (lookup_reference_data) before the correct number is even
knowable — a model that skips the lookup and guesses will generally be
wrong, which is exactly the grounding behavior the benchmark is checking
alongside raw arithmetic correctness.
"""
from dataclasses import dataclass


@dataclass
class ReasoningCase:
    name: str
    question: str
    expected_answer_substrings: list[str]  # ANY of these appearing in the answer counts as correct
    note: str = ""


CASES: list[ReasoningCase] = [
    ReasoningCase(
        name="simple_percentage",
        question="What is 15% of 200?",
        expected_answer_substrings=["30"],
        note="Trivial — a good check that low effort isn't wasted here.",
    ),
    ReasoningCase(
        name="single_hop_gap_percentage",
        question=(
            "By what percentage would Marketing's Q3 revenue need to grow "
            "to cross its alert threshold?"
        ),
        # (550000-482000)/482000*100 = 14.106...%
        expected_answer_substrings=["14.1", "14.0", "14%", "~14"],
        note="Requires one lookup + one calculation, in that order.",
    ),
    ReasoningCase(
        name="multi_hop_projection",
        question=(
            "If Sales grows 10% per quarter for the next two quarters "
            "starting from its Q3 figure, will it cross its alert "
            "threshold, and by how much?"
        ),
        # 1,240,000 * 1.1^2 = 1,500,400 vs threshold 1,500,000 -> crosses by 400
        expected_answer_substrings=["400", "1,500,400", "1500400"],
        note="Genuinely multi-step: lookup, then a compounding calculation, then a comparison.",
    ),
    ReasoningCase(
        name="threshold_ratio",
        question="What percentage above its Q3 value is Engineering Cloud Spend's alert threshold?",
        # (120000-96000)/96000*100 = 25%
        expected_answer_substrings=["25%", "25.0"],
        note="",
    ),
]
