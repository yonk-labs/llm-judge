from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from .io import dump_jsonl
from .models import EvalCase, JudgeDecision, ReferenceDecision, normalize_verdict
from .providers import LLMProvider
from .report import result_row, write_case_audit, write_summary
from .scorers import generate_answer, generate_reference, llm_score, quick_score


def evaluate_case(
    case: EvalCase,
    *,
    mode: str,
    synonyms: dict[str, list[str]],
    judge_provider: LLMProvider | None = None,
    judge_providers: Sequence[LLMProvider] | None = None,
    answer_provider: LLMProvider | None = None,
    reference_provider: LLMProvider | None = None,
    reference_providers: Sequence[LLMProvider] | None = None,
    generate_missing_answer: bool = False,
    generate_missing_expected: bool = False,
    llm_threshold: float = 0.72,
    parse_retries: int = 1,
) -> tuple[EvalCase, JudgeDecision]:
    """Generate and/or judge one case.

    Parameters
    ----------
    case:
        Normalized input case.
    mode:
        ``quick``, ``accurate``, or ``dual``.
    synonyms:
        Synonym map used by quick scoring.
    judge_provider:
        Provider used by accurate/dual LLM judging.
    judge_providers:
        Optional ensemble of one to three providers. When supplied, this takes
        precedence over ``judge_provider``.
    answer_provider:
        Provider used when ``generate_missing_answer`` is enabled.
    reference_provider:
        Provider used when ``generate_missing_expected`` is enabled.
    reference_providers:
        Optional ensemble of one to three providers for reference/gold generation.
    generate_missing_answer:
        If true, generate an answer when ``case.answer`` is empty.
    llm_threshold:
        In dual mode, quick scores below this threshold are sent to the LLM judge.
    parse_retries:
        Number of malformed judge-JSON repair attempts.
    """
    working = case
    if generate_missing_expected and not working.expected.strip():
        providers = _resolve_reference_providers(reference_provider, reference_providers)
        if not providers:
            return working, JudgeDecision.error("reference generation requested but no reference provider configured")
        reference = generate_reference_with_providers(working, providers, parse_retries=parse_retries)
        working = replace(
            working,
            expected=reference.expected,
            expected_facts=reference.expected_facts or working.expected_facts,
            metadata={
                **working.metadata,
                "acceptable_answers": reference.acceptable_answers,
                "reference_generation": {
                    "provider": reference.provider,
                    "model": reference.model,
                    "latency_ms": reference.latency_ms,
                    "error": reference.error,
                    "rationale": reference.rationale,
                    "acceptable_answers": reference.acceptable_answers,
                    "raw": reference.raw,
                },
            },
        )
        if reference.error:
            return working, JudgeDecision.error(
                f"reference generation failed: {reference.error}",
                provider=reference.provider,
                model=reference.model,
                latency_ms=reference.latency_ms,
            )

    if generate_missing_answer and not working.answer.strip():
        if answer_provider is None:
            return working, JudgeDecision.error("answer generation requested but no answer provider configured")
        answer = generate_answer(working, answer_provider)
        working = replace(
            working,
            answer=answer.answer,
            metadata={
                **working.metadata,
                "generated_answer": answer.answer,
                "answer_generation": {
                    "provider": answer.provider,
                    "model": answer.model,
                    "latency_ms": answer.latency_ms,
                    "error": answer.error,
                },
            },
        )
        if answer.error:
            return working, JudgeDecision.error(
                f"answer generation failed: {answer.error}",
                provider=answer.provider,
                model=answer.model,
                latency_ms=answer.latency_ms,
            )

    decision = quick_score(working, synonyms=synonyms)
    if mode == "accurate":
        providers = _resolve_judge_providers(judge_provider, judge_providers)
        if not providers:
            return working, JudgeDecision.error("accurate mode requested but no judge provider configured")
        decision = score_with_judges(working, providers, parse_retries=parse_retries)
    elif mode == "dual" and decision.score < llm_threshold:
        providers = _resolve_judge_providers(judge_provider, judge_providers)
        if providers:
            decision = score_with_judges(working, providers, parse_retries=parse_retries)
    return working, decision


def _resolve_judge_providers(
    judge_provider: LLMProvider | None,
    judge_providers: Sequence[LLMProvider] | None,
) -> list[LLMProvider]:
    if judge_providers is not None:
        return list(judge_providers)
    return [judge_provider] if judge_provider is not None else []


def _resolve_reference_providers(
    reference_provider: LLMProvider | None,
    reference_providers: Sequence[LLMProvider] | None,
) -> list[LLMProvider]:
    if reference_providers is not None:
        return list(reference_providers)
    return [reference_provider] if reference_provider is not None else []


def generate_reference_with_providers(
    case: EvalCase,
    providers: Sequence[LLMProvider],
    *,
    parse_retries: int = 1,
) -> ReferenceDecision:
    """Generate a reference answer with one to three providers and keep variants."""
    if len(providers) > 3:
        return ReferenceDecision(expected="", error="at most 3 reference providers are supported")
    decisions = [generate_reference(case, provider, parse_retries=parse_retries) for provider in providers]
    valid = [decision for decision in decisions if not decision.error and decision.expected.strip()]
    if not valid:
        return ReferenceDecision(
            expected="",
            rationale="all reference generators failed",
            provider="reference-ensemble",
            model=",".join(decision.model for decision in decisions),
            latency_ms=sum(decision.latency_ms for decision in decisions),
            raw={"individual_references": [_reference_raw(decision) for decision in decisions]},
            error="all reference generators failed",
        )
    primary = valid[0]
    acceptable = []
    for decision in valid:
        acceptable.extend([decision.expected, *decision.acceptable_answers])
    deduped_acceptable = _dedupe(acceptable)
    rationale = primary.rationale
    if len(decisions) > 1:
        rationale = f"Primary reference from {primary.provider}/{primary.model}. Collected {len(valid)}/{len(decisions)} successful reference generations."
    return ReferenceDecision(
        expected=primary.expected,
        expected_facts=primary.expected_facts,
        acceptable_answers=deduped_acceptable,
        rationale=rationale,
        latency_ms=sum(decision.latency_ms for decision in decisions),
        provider=primary.provider if len(decisions) == 1 else "reference-ensemble",
        model=primary.model if len(decisions) == 1 else ",".join(decision.model for decision in decisions),
        raw={"individual_references": [_reference_raw(decision) for decision in decisions]},
    )


def _reference_raw(decision: ReferenceDecision) -> dict:
    return {
        "provider": decision.provider,
        "model": decision.model,
        "expected": decision.expected,
        "expected_facts": decision.expected_facts,
        "acceptable_answers": decision.acceptable_answers,
        "rationale": decision.rationale,
        "latency_ms": decision.latency_ms,
        "error": decision.error,
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return result


def score_with_judges(
    case: EvalCase,
    providers: Sequence[LLMProvider],
    *,
    parse_retries: int = 1,
) -> JudgeDecision:
    """Judge a case with one to three providers and aggregate the decisions."""
    if len(providers) > 3:
        return JudgeDecision.error("at most 3 judge providers are supported")
    decisions = [llm_score(case, provider, parse_retries=parse_retries) for provider in providers]
    if len(decisions) == 1:
        return decisions[0]

    valid = [decision for decision in decisions if normalize_verdict(decision.verdict) != "ERROR"]
    raw = {
        "individual_judges": [
            {
                "provider": decision.provider,
                "model": decision.model,
                "verdict": decision.verdict,
                "score": decision.score,
                "rationale": decision.rationale,
                "latency_ms": decision.latency_ms,
                "missing": decision.missing,
                "supported": decision.supported,
                "contradictions": decision.contradictions,
                "raw": decision.raw,
            }
            for decision in decisions
        ]
    }
    if not valid:
        return JudgeDecision(
            score=0.0,
            verdict="ERROR",
            rationale="all judges failed",
            raw=raw,
            provider="ensemble",
            model=",".join(decision.model for decision in decisions),
            latency_ms=sum(decision.latency_ms for decision in decisions),
        )

    verdict_counts = Counter(normalize_verdict(decision.verdict) for decision in valid)
    top_verdict, top_count = verdict_counts.most_common(1)[0]
    avg_score = sum(decision.score for decision in valid) / len(valid)
    if top_count > len(valid) / 2:
        verdict = top_verdict
    elif avg_score >= 0.82:
        verdict = "CORRECT"
    elif avg_score >= 0.58:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"

    supported = sorted({item for decision in valid for item in decision.supported})
    missing = sorted({item for decision in valid for item in decision.missing})
    contradictions = sorted({item for decision in valid for item in decision.contradictions})
    rationale = (
        f"Aggregated {len(valid)}/{len(decisions)} successful judges. "
        f"Verdict counts: {dict(verdict_counts)}. Average score: {avg_score:.3f}."
    )
    return JudgeDecision(
        score=round(avg_score, 3),
        verdict=verdict,
        rationale=rationale,
        missing=missing,
        supported=supported,
        contradictions=contradictions,
        retrieval_score=_avg_optional([decision.retrieval_score for decision in valid]),
        answer_score=_avg_optional([decision.answer_score for decision in valid]),
        raw=raw,
        latency_ms=sum(decision.latency_ms for decision in decisions),
        provider="ensemble",
        model=",".join(decision.model for decision in decisions),
    )


def _avg_optional(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 3)


def evaluate_cases(
    cases: Iterable[EvalCase],
    *,
    out_dir: Path,
    mode: str,
    synonyms: dict[str, list[str]],
    judge_provider: LLMProvider | None = None,
    judge_providers: Sequence[LLMProvider] | None = None,
    answer_provider: LLMProvider | None = None,
    reference_provider: LLMProvider | None = None,
    reference_providers: Sequence[LLMProvider] | None = None,
    generate_missing_answer: bool = False,
    generate_missing_expected: bool = False,
    concurrency: int = 1,
    llm_threshold: float = 0.72,
    parse_retries: int = 1,
    resume: bool = False,
) -> list[dict]:
    """Evaluate many cases with optional concurrency, cache-backed providers, and resume.

    Writes ``results.jsonl``, ``summary.md``, and per-case audits incrementally.

    Parameters
    ----------
    cases:
        Cases to process.
    out_dir:
        Output directory for results and audits.
    mode:
        ``quick``, ``accurate``, or ``dual``.
    synonyms:
        Synonym map used by quick scoring.
    judge_provider:
        Provider used by accurate/dual LLM judging.
    judge_providers:
        Optional ensemble of one to three judge providers.
    answer_provider:
        Provider used for answer generation.
    reference_provider:
        Provider used for reference/gold answer generation.
    reference_providers:
        Optional ensemble of one to three reference/gold generators.
    generate_missing_answer:
        Generate answers for cases where ``answer`` is empty.
    concurrency:
        Number of worker threads. Use 1 for serial deterministic execution.
    llm_threshold:
        Quick-score threshold for dual mode.
    parse_retries:
        Judge JSON repair retry count.
    resume:
        If true, skip case IDs already present in ``results.jsonl``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "cases"
    rows_by_id: dict[str, dict] = {}
    results_path = out_dir / "results.jsonl"
    if resume and results_path.exists():
        import json

        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_id[str(row["id"])] = row

    pending = [case for case in cases if str(case.case_id) not in rows_by_id]

    def _run(case: EvalCase) -> tuple[str, dict]:
        judged_case, decision = evaluate_case(
            case,
            mode=mode,
            synonyms=synonyms,
            judge_provider=judge_provider,
            judge_providers=judge_providers,
            answer_provider=answer_provider,
            reference_provider=reference_provider,
            reference_providers=reference_providers,
            generate_missing_answer=generate_missing_answer,
            generate_missing_expected=generate_missing_expected,
            llm_threshold=llm_threshold,
            parse_retries=parse_retries,
        )
        write_case_audit(case_dir, judged_case, decision)
        return judged_case.case_id, result_row(judged_case, decision)

    if concurrency <= 1:
        for case in pending:
            case_id, row = _run(case)
            rows_by_id[str(case_id)] = row
            dump_jsonl(results_path, rows_by_id.values())
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run, case) for case in pending]
            for future in as_completed(futures):
                case_id, row = future.result()
                rows_by_id[str(case_id)] = row
                dump_jsonl(results_path, rows_by_id.values())

    rows = list(rows_by_id.values())
    write_summary(out_dir, rows)
    return rows
