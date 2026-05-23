from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VERDICT_ALIASES = {
    "WRONG": "INCORRECT",
    "FAIL": "INCORRECT",
    "FAILED": "INCORRECT",
    "FULLY_CORRECT": "CORRECT",
    "PARTIALLY_CORRECT": "PARTIAL",
    "PARTLY_CORRECT": "PARTIAL",
}


def normalize_verdict(verdict: str) -> str:
    """Return the canonical verdict name for a provider or legacy verdict string.

    Parameters
    ----------
    verdict:
        Raw verdict value such as ``CORRECT``, ``WRONG``, ``FAIL``, or
        ``partially_correct``.
    """
    normalized = verdict.strip().upper().replace("-", "_").replace(" ", "_")
    return VERDICT_ALIASES.get(normalized, normalized)


@dataclass
class EvalCase:
    """One benchmark case after input-profile normalization.

    Parameters
    ----------
    case_id:
        Stable case identifier used for result rows and audit file names.
    question:
        User/query text being evaluated.
    answer:
        Generated answer to judge. May be empty when answer generation is enabled.
    expected:
        Gold/reference answer.
    expected_facts:
        Optional required facts that should be covered by the answer.
    chunks:
        Retrieved chunks, summaries, or evidence supplied to the answer/judge.
    reference_contexts:
        Full/oracle context used to synthesize a missing gold/reference answer.
        Falls back to ``chunks`` when empty.
    settings:
        Benchmark settings that should appear in audits, such as mode/config labels.
    metadata:
        Extra source fields preserved but not interpreted by the judge.
    source_record:
        Original input record before profile normalization. Used for audit/replay.
    """
    case_id: str
    question: str
    answer: str
    expected: str
    expected_facts: list[str] = field(default_factory=list)
    chunks: list[Any] = field(default_factory=list)
    reference_contexts: list[Any] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerDecision:
    """Result of the optional answer-generation step."""
    answer: str
    rationale: str = ""
    latency_ms: int = 0
    provider: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ReferenceDecision:
    """Result of generating a missing gold/reference answer from oracle context."""
    expected: str
    expected_facts: list[str] = field(default_factory=list)
    acceptable_answers: list[str] = field(default_factory=list)
    rationale: str = ""
    latency_ms: int = 0
    provider: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class JudgeDecision:
    """Normalized result of a quick or LLM-based judge decision."""
    score: float
    verdict: str
    rationale: str
    missing: list[str] = field(default_factory=list)
    supported: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    retrieval_score: float | None = None
    answer_score: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    provider: str = "quick"
    model: str = "heuristic"

    @property
    def passed(self) -> bool:
        """Whether this decision counts as passing under the default threshold."""
        return normalize_verdict(self.verdict) in {"CORRECT", "PARTIAL"} and self.score >= 0.72

    @classmethod
    def error(cls, message: str, *, provider: str = "", model: str = "", latency_ms: int = 0) -> "JudgeDecision":
        """Build a non-fatal ``ERROR`` decision for provider/runtime failures."""
        return cls(
            score=0.0,
            verdict="ERROR",
            rationale=message,
            raw={"error": message},
            latency_ms=latency_ms,
            provider=provider,
            model=model,
        )
