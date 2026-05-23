from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .models import AnswerDecision, EvalCase, JudgeDecision, normalize_verdict
from .providers import LLMProvider

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

BUILTIN_SYNONYMS = {
    "usa": ["united states", "us", "u.s."],
    "us": ["united states", "usa", "u.s."],
    "ai": ["artificial intelligence"],
    "llm": ["large language model"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
}


def normalize(text: str) -> str:
    """Normalize text for lightweight lexical comparison."""
    return re.sub(r"\s+", " ", text.casefold().replace("&", " and ")).strip()


def tokens(text: str) -> list[str]:
    """Tokenize text for heuristic scoring."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalize(text))
        if len(token) > 1 and token not in STOPWORDS
    ]


def phrase_candidates(text: str) -> list[str]:
    """Extract likely required phrases from an expected answer."""
    normalized = normalize(text)
    phrases = set()
    for quoted in re.findall(r'"([^"]+)"|`([^`]+)`', text):
        phrase = quoted[0] or quoted[1]
        if phrase.strip():
            phrases.add(normalize(phrase))
    for match in re.findall(r"\b([A-Z][\w.-]*(?:\s+(?:v\.|vs\.|of|the|and|[A-Z][\w.-]*)){1,8})", text):
        phrases.add(normalize(match))
    words = tokens(text)
    for width in (4, 3, 2):
        for i in range(0, max(0, len(words) - width + 1)):
            phrases.add(" ".join(words[i : i + width]))
    return sorted(phrases, key=lambda item: (-len(item.split()), item))[:30]


def load_synonyms(path: str | None) -> dict[str, list[str]]:
    """Load built-in plus optional user synonyms from a JSON file."""
    synonyms = {key: list(value) for key, value in BUILTIN_SYNONYMS.items()}
    if not path:
        return synonyms
    with open(path, "r", encoding="utf-8") as handle:
        user_map = json.load(handle)
    for key, values in user_map.items():
        synonyms[normalize(key)] = [normalize(str(value)) for value in values]
    return synonyms


def _term_present(term: str, haystack: str, synonyms: dict[str, list[str]]) -> bool:
    term = normalize(term)
    haystack = normalize(haystack)
    if term and term in haystack:
        return True
    compact_term = re.sub(r"[^a-z0-9]", "", term)
    compact_haystack = re.sub(r"[^a-z0-9]", "", haystack)
    if compact_term and compact_term in compact_haystack:
        return True
    for alt in synonyms.get(term, []):
        if normalize(alt) in haystack:
            return True
    term_tokens = tokens(term)
    if not term_tokens:
        return False
    hay_tokens = set(tokens(haystack))
    if len(term_tokens) >= 2 and len(term_tokens[0]) >= 5 and term_tokens[0] in hay_tokens:
        return True
    token_hit_rate = sum(1 for token in term_tokens if token in hay_tokens) / len(term_tokens)
    if len(term_tokens) <= 2:
        return token_hit_rate == 1.0
    return token_hit_rate >= 0.67


def cosine_similarity(left: list[str], right: list[str]) -> float:
    """Return cosine similarity between two token lists."""
    if not left or not right:
        return 0.0
    a = Counter(left)
    b = Counter(right)
    shared = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in a.values()))
    right_norm = math.sqrt(sum(value * value for value in b.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def quick_score(case: EvalCase, synonyms: dict[str, list[str]] | None = None) -> JudgeDecision:
    """Score an answer with deterministic, paraphrase-tolerant heuristics."""
    start = time.perf_counter()
    synonyms = synonyms or BUILTIN_SYNONYMS
    expected_tokens = tokens(case.expected)
    answer_tokens = tokens(case.answer)
    chunk_text = "\n".join(str(chunk) for chunk in case.chunks)
    chunk_tokens = tokens(chunk_text)
    expected_phrases = phrase_candidates(case.expected)

    supported = [phrase for phrase in expected_phrases if _term_present(phrase, case.answer, synonyms)]
    missing = [phrase for phrase in expected_phrases[:12] if phrase not in supported]
    answer_token_set = set(answer_tokens)
    lead_alias_score = 0.0
    for phrase in expected_phrases:
        phrase_tokens = tokens(phrase)
        if len(phrase_tokens) >= 2 and len(phrase_tokens[0]) >= 5 and phrase_tokens[0] in answer_token_set:
            lead_alias_score = 0.62
            break

    token_coverage = 0.0
    if expected_tokens:
        token_coverage = sum(1 for token in set(expected_tokens) if token in answer_token_set) / len(set(expected_tokens))
    semantic_overlap = cosine_similarity(expected_tokens, answer_tokens)
    fuzzy = SequenceMatcher(None, normalize(case.expected), normalize(case.answer)).ratio()
    phrase_score = len(supported) / max(1, min(12, len(expected_phrases)))
    retrieval_score = cosine_similarity(expected_tokens, chunk_tokens)
    if lead_alias_score and retrieval_score >= 0.55:
        lead_alias_score = 0.82
    score = max(
        token_coverage,
        0.65 * semantic_overlap + 0.35 * fuzzy,
        0.55 * phrase_score + 0.45 * token_coverage,
        lead_alias_score,
    )
    if score >= 0.82:
        verdict = "CORRECT"
    elif score >= 0.58:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"
    if verdict == "CORRECT":
        missing = []
    latency_ms = int((time.perf_counter() - start) * 1000)
    rationale = (
        f"Quick scorer used token coverage={token_coverage:.2f}, "
        f"semantic overlap={semantic_overlap:.2f}, fuzzy={fuzzy:.2f}, "
        f"phrase support={phrase_score:.2f}. It is paraphrase-tolerant but not a substitute for a calibrated LLM judge."
    )
    return JudgeDecision(
        score=round(score, 3),
        verdict=verdict,
        rationale=rationale,
        missing=missing,
        supported=supported[:12],
        retrieval_score=round(retrieval_score, 3),
        answer_score=round(score, 3),
        latency_ms=latency_ms,
    )


def judge_prompt(case: EvalCase) -> str:
    """Build the strict-JSON LLM judge prompt for a case."""
    chunks = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(case.chunks))
    settings = json.dumps(case.settings, ensure_ascii=False, indent=2, sort_keys=True)
    expected_facts = "\n".join(f"- {fact}" for fact in case.expected_facts) or "None supplied."
    return f"""You are judging a RAG benchmark answer. Be fair and paraphrase-tolerant.

Do not require exact wording. Treat abbreviations, aliases, synonyms, reordered facts, and partial case names as equivalent when they clearly identify the same fact. Example: "Bostock" can satisfy "Bostock v. Clayton County" if the context makes the reference unambiguous.

Grade against the expected answer and the retrieved chunks. Penalize hallucinated contradictions. Do not penalize missing nice-to-have details unless they are necessary to answer the question.

Return only JSON with this schema:
{{
  "verdict": "CORRECT|PARTIAL|INCORRECT",
  "score": 0.0,
  "answer_score": 0.0,
  "retrieval_score": 0.0,
  "supported": ["facts the answer got right"],
  "missing": ["important expected facts not present"],
  "contradictions": ["facts contradicted by the answer"],
  "rationale": "short explanation"
}}

Question:
{case.question}

Expected answer:
{case.expected}

Expected required facts:
{expected_facts}

Retrieved chunks:
{chunks}

Settings:
{settings}

Answer to judge:
{case.answer}
"""


def answer_prompt(case: EvalCase) -> str:
    """Build the retrieved-context answer-generation prompt for a case."""
    chunks = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(case.chunks))
    settings = json.dumps(case.settings, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""Answer the question using only the retrieved context.

If the context is insufficient, say "Insufficient information." Keep the answer concise but complete.

Question:
{case.question}

Retrieved context:
{chunks}

Settings:
{settings}
"""


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def generate_answer(case: EvalCase, provider: LLMProvider) -> AnswerDecision:
    """Generate an answer from retrieved chunks using plain text provider mode."""
    start = time.perf_counter()
    try:
        response = provider.complete(answer_prompt(case), json_mode=False)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return AnswerDecision(
            answer="",
            rationale=f"answer generation failed: {exc}",
            latency_ms=latency_ms,
            provider=provider.name,
            model=provider.model,
            error=str(exc),
        )
    return AnswerDecision(
        answer=response.text.strip(),
        latency_ms=response.latency_ms or int((time.perf_counter() - start) * 1000),
        provider=provider.name,
        model=provider.model,
        raw={"usage": response.usage},
    )


def llm_score(case: EvalCase, provider: LLMProvider, *, parse_retries: int = 1) -> JudgeDecision:
    """Judge a case with an LLM provider and return a non-fatal decision."""
    start = time.perf_counter()
    prompt = judge_prompt(case)
    last_text = ""
    try:
        for attempt in range(parse_retries + 1):
            response = provider.complete(prompt, json_mode=True)
            last_text = response.text
            try:
                parsed = _extract_json(response.text)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt >= parse_retries:
                    raise exc
                prompt = (
                    "Return only valid JSON matching the requested judge schema. "
                    "Do not include markdown fences or commentary.\n\n"
                    f"Invalid prior response:\n{response.text}\n\nOriginal judging task:\n{judge_prompt(case)}"
                )
        else:
            parsed = {}
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return JudgeDecision.error(
            f"judge failed: {exc}",
            provider=provider.name,
            model=provider.model,
            latency_ms=latency_ms,
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError) as exc:
        return JudgeDecision.error(
            f"judge returned invalid score: {exc}",
            provider=provider.name,
            model=provider.model,
            latency_ms=latency_ms,
        )
    verdict = normalize_verdict(str(parsed.get("verdict", "INCORRECT")))
    return JudgeDecision(
        score=round(max(0.0, min(1.0, score)), 3),
        verdict=verdict,
        rationale=str(parsed.get("rationale", "")),
        missing=[str(item) for item in parsed.get("missing", [])],
        supported=[str(item) for item in parsed.get("supported", [])],
        contradictions=[str(item) for item in parsed.get("contradictions", [])],
        retrieval_score=_safe_float(parsed.get("retrieval_score")),
        answer_score=_safe_float(parsed.get("answer_score")),
        raw={"judge_json": parsed, "usage": response.usage, "raw_text": last_text},
        latency_ms=response.latency_ms or latency_ms,
        provider=provider.name,
        model=provider.model,
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
