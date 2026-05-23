from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .models import AnswerDecision, EvalCase, JudgeDecision, ReferenceDecision, normalize_verdict
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
    if case.expected_facts:
        decision = _quick_score_required_facts(case, synonyms)
        decision.latency_ms = int((time.perf_counter() - start) * 1000)
        return decision
    acceptable = case.metadata.get("acceptable_answers") or case.metadata.get("reference_generation", {}).get("acceptable_answers", [])
    variants = [case.expected, *[str(item) for item in acceptable]]
    decisions = [_quick_score_variant(case, expected, synonyms) for expected in variants if str(expected).strip()]
    if not decisions:
        return JudgeDecision.error("no expected answer available for quick scoring")
    decision = max(decisions, key=lambda item: item.score)
    decision.latency_ms = int((time.perf_counter() - start) * 1000)
    if decision.raw.get("expected_variant") and decision.raw["expected_variant"] != case.expected:
        decision.rationale = f"Matched acceptable answer variant: {decision.raw['expected_variant']}. {decision.rationale}"
    return decision


def _quick_score_required_facts(case: EvalCase, synonyms: dict[str, list[str]]) -> JudgeDecision:
    supported = [fact for fact in case.expected_facts if _fact_present(fact, case.answer, synonyms)]
    missing = [fact for fact in case.expected_facts if fact not in supported]
    score = len(supported) / len(case.expected_facts)
    if score >= 0.999:
        verdict = "CORRECT"
        missing = []
    elif score > 0:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"
    chunk_text = "\n".join(str(chunk) for chunk in case.chunks)
    fact_tokens = tokens(" ".join(case.expected_facts))
    retrieval_score = cosine_similarity(fact_tokens, tokens(chunk_text))
    rationale = (
        f"Quick scorer used required-fact coverage={score:.2f} "
        f"({len(supported)}/{len(case.expected_facts)} facts supported). "
        "When expected_facts are supplied, they define the grading units."
    )
    return JudgeDecision(
        score=round(score, 3),
        verdict=verdict,
        rationale=rationale,
        missing=missing,
        supported=supported,
        retrieval_score=round(retrieval_score, 3),
        answer_score=round(score, 3),
        raw={"fact_coverage": score, "required_fact_count": len(case.expected_facts)},
    )


def _fact_present(fact: str, answer: str, synonyms: dict[str, list[str]]) -> bool:
    if _term_present(fact, answer, synonyms):
        return True
    for candidate in _fact_value_candidates(fact):
        if candidate and _term_present(candidate, answer, synonyms):
            return True
    return False


def _fact_value_candidates(fact: str) -> list[str]:
    candidates = []
    for separator in (":", "=", " is ", " was ", " are ", " were "):
        if separator in fact:
            value = fact.split(separator, 1)[1].strip(" .;")
            if value:
                candidates.append(value)
    return candidates


def _quick_score_variant(case: EvalCase, expected: str, synonyms: dict[str, list[str]]) -> JudgeDecision:
    expected_tokens = tokens(expected)
    answer_tokens = tokens(case.answer)
    chunk_text = "\n".join(str(chunk) for chunk in case.chunks)
    chunk_tokens = tokens(chunk_text)
    expected_phrases = phrase_candidates(expected)

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
    fuzzy = SequenceMatcher(None, normalize(expected), normalize(case.answer)).ratio()
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
        raw={"expected_variant": expected},
    )


def judge_prompt(case: EvalCase) -> str:
    """Build the strict-JSON LLM judge prompt for a case."""
    chunks = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(case.chunks))
    settings = json.dumps(case.settings, ensure_ascii=False, indent=2, sort_keys=True)
    expected_facts = "\n".join(f"- {fact}" for fact in case.expected_facts) or "None supplied."
    acceptable = case.metadata.get("acceptable_answers") or case.metadata.get("reference_generation", {}).get("acceptable_answers", [])
    acceptable_answers = "\n".join(f"- {item}" for item in acceptable) or "None supplied."
    return f"""You are judging a RAG benchmark answer. Be fair and paraphrase-tolerant.

Do not require exact wording. Treat abbreviations, aliases, synonyms, reordered facts, and partial case names as equivalent when they clearly identify the same fact. Example: "Bostock" can satisfy "Bostock v. Clayton County" if the context makes the reference unambiguous.

Grade against the question, expected answer, expected required facts, acceptable answers, and retrieved chunks. Penalize hallucinated contradictions. Do not penalize missing nice-to-have details unless they are necessary to answer the question.

Use the granularity requested by the question. If the question broadly asks "where" and no stricter required facts are listed, an answer may be fully correct when it gives any true state, city, or specific site that answers the location. If the question asks for specific fields such as "city and hospital", require those fields. Required facts override this broad-location leniency.

For multi-part questions, assign partial credit by required fact coverage. If the answer satisfies one of two required facts, it should usually be PARTIAL with score near 0.50. If it satisfies two of three required facts, score near 0.67. Do not mark a partial answer fully correct because it mentions one correct detail.

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

Acceptable answers or aliases:
{acceptable_answers}

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


def reference_prompt(case: EvalCase) -> str:
    """Build the oracle-context prompt used to synthesize missing reference answers."""
    source = case.reference_contexts or case.chunks
    contexts = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(source))
    settings = json.dumps(case.settings, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""Create a benchmark reference answer from the full/oracle context.

Return only JSON with this schema:
{{
  "expected": "concise reference answer",
  "expected_facts": ["facts that are strictly required by the question"],
  "acceptable_answers": ["semantically acceptable shorter or alternate answers"],
  "rationale": "short explanation of answer granularity"
}}

Rules:
- Use only the supplied full/oracle context.
- Match the question's requested granularity.
- Do not make every detail in the context mandatory.
- If the question asks broadly, list broader true answers as acceptable aliases and do not create strict required facts for every supported granularity.
- If the question asks for specific fields, make those fields required facts.
- For multi-part questions, split each requested field into its own required fact so partial answers can be scored proportionally. Example: for "city and hospital", use separate facts for city and hospital.
- For broad questions where several granularities are acceptable, prefer `acceptable_answers` over required facts.
- If the context is insufficient, set expected to "Insufficient information." and explain why.

Question:
{case.question}

Full/oracle context:
{contexts}

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


def generate_reference(case: EvalCase, provider: LLMProvider, *, parse_retries: int = 1) -> ReferenceDecision:
    """Generate a missing gold/reference answer from full/oracle context."""
    start = time.perf_counter()
    prompt = reference_prompt(case)
    try:
        for attempt in range(parse_retries + 1):
            response = provider.complete(prompt, json_mode=True)
            try:
                parsed = _extract_json(response.text)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt >= parse_retries:
                    raise exc
                prompt = (
                    "Return only valid JSON matching the requested reference-answer schema. "
                    "Do not include markdown fences or commentary.\n\n"
                    f"Invalid prior response:\n{response.text}\n\nOriginal task:\n{reference_prompt(case)}"
                )
        else:
            parsed = {}
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ReferenceDecision(
            expected="",
            rationale=f"reference generation failed: {exc}",
            latency_ms=latency_ms,
            provider=provider.name,
            model=provider.model,
            error=str(exc),
        )
    latency_ms = int((time.perf_counter() - start) * 1000)
    expected = str(parsed.get("expected") or "").strip()
    facts = parsed.get("expected_facts", [])
    acceptable = parsed.get("acceptable_answers", [])
    return ReferenceDecision(
        expected=expected,
        expected_facts=[str(item) for item in facts] if isinstance(facts, list) else [str(facts)],
        acceptable_answers=[str(item) for item in acceptable] if isinstance(acceptable, list) else [str(acceptable)],
        rationale=str(parsed.get("rationale") or ""),
        latency_ms=latency_ms,
        provider=provider.name,
        model=provider.model,
        raw={"usage": getattr(response, "usage", {})},
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
