from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import EvalCase, JudgeDecision, normalize_verdict
from .scorers import answer_prompt, judge_prompt, reference_prompt


def safe_name(value: str) -> str:
    """Return a filesystem-safe case file name stem."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return slug or "case"


def write_case_audit(case_dir: Path, case: EvalCase, decision: JudgeDecision) -> Path:
    """Write one Markdown audit file for a judged case."""
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{safe_name(case.case_id)}.md"
    chunks = "\n\n".join(f"### Chunk {index + 1}\n\n{chunk}" for index, chunk in enumerate(case.chunks))
    body = f"""# Judge Audit - {case.case_id}

## Verdict

- Verdict: {decision.verdict}
- Score: {decision.score:.3f}
- Provider: {decision.provider}
- Model: {decision.model}
- Judge latency: {decision.latency_ms} ms
- Answer score: {decision.answer_score}
- Retrieval score: {decision.retrieval_score}

## Question

{case.question}

## Settings

```json
{json.dumps(case.settings, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Retrieved Chunks

{chunks or "No chunks supplied."}

## LLM Answer

{case.answer}

## Expected Answer

{case.expected}

## Expected Required Facts

{_list(case.expected_facts)}

## Acceptable Answers

{_list([str(item) for item in case.metadata.get("acceptable_answers", [])])}

## Judge Rationale

{decision.rationale}

## Supported Facts

{_list(decision.supported)}

## Missing Facts

{_list(decision.missing)}

## Contradictions

{_list(decision.contradictions)}

## Metadata

```json
{json.dumps(case.metadata, indent=2, ensure_ascii=False, sort_keys=True)}
```
"""
    path.write_text(body, encoding="utf-8")
    return path


def write_trace_audit(audit_dir: Path, case: EvalCase, decision: JudgeDecision) -> Path:
    """Write replay-oriented audit files for one judged case."""
    case_dir = audit_dir / safe_name(case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)

    normalized_case = {
        "source_record": case.source_record,
        "id": case.case_id,
        "question": case.question,
        "answer": case.answer,
        "expected": case.expected,
        "expected_facts": case.expected_facts,
        "chunks": case.chunks,
        "reference_contexts": case.reference_contexts,
        "settings": case.settings,
        "metadata": case.metadata,
    }
    prompts = {
        "answer": answer_prompt(case),
        "reference": reference_prompt(case),
        "judge": judge_prompt(case),
    }
    raw = {
        "decision": {
            "verdict": decision.verdict,
            "score": decision.score,
            "passed": decision.passed,
            "provider": decision.provider,
            "model": decision.model,
            "latency_ms": decision.latency_ms,
            "answer_score": decision.answer_score,
            "retrieval_score": decision.retrieval_score,
            "supported": decision.supported,
            "missing": decision.missing,
            "contradictions": decision.contradictions,
            "rationale": decision.rationale,
            "raw": decision.raw,
        },
        "metadata": case.metadata,
    }

    _write_json(case_dir / "case.json", normalized_case)
    _write_json(case_dir / "chunks.json", {"chunks": case.chunks, "reference_contexts": case.reference_contexts})
    _write_json(case_dir / "prompts.json", prompts)
    _write_json(case_dir / "raw.json", raw)
    (case_dir / "prompt-answer.txt").write_text(prompts["answer"], encoding="utf-8")
    (case_dir / "prompt-reference.txt").write_text(prompts["reference"], encoding="utf-8")
    (case_dir / "prompt-judge.txt").write_text(prompts["judge"], encoding="utf-8")
    return case_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _list(values: list[str]) -> str:
    if not values:
        return "- None"
    return "\n".join(f"- {value}" for value in values)


def result_row(case: EvalCase, decision: JudgeDecision) -> dict[str, Any]:
    """Convert a judged case to a machine-readable result row."""
    return {
        "id": case.case_id,
        "question": case.question,
        "answer": case.answer,
        "expected": case.expected,
        "expected_facts": case.expected_facts,
        "reference_contexts": case.reference_contexts,
        "verdict": decision.verdict,
        "score": decision.score,
        "passed": decision.passed,
        "latency_ms": decision.latency_ms,
        "provider": decision.provider,
        "model": decision.model,
        "answer_score": decision.answer_score,
        "retrieval_score": decision.retrieval_score,
        "supported": decision.supported,
        "missing": decision.missing,
        "contradictions": decision.contradictions,
        "rationale": decision.rationale,
        "raw": decision.raw,
        "settings": case.settings,
        "metadata": case.metadata,
    }


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write aggregate ``summary.md`` for a run."""
    total = len(rows)
    correct = sum(1 for row in rows if normalize_verdict(row["verdict"]) == "CORRECT")
    partial = sum(1 for row in rows if normalize_verdict(row["verdict"]) == "PARTIAL")
    errors = sum(1 for row in rows if normalize_verdict(row["verdict"]) == "ERROR")
    passed = sum(1 for row in rows if row["passed"])
    avg_score = sum(float(row["score"]) for row in rows) / total if total else 0.0
    avg_latency = sum(int(row["latency_ms"]) for row in rows) / total if total else 0.0
    lines = [
        "# LLM Judge Summary",
        "",
        "## Run Metrics",
        "",
        f"- Total cases: {total}",
        f"- Correct: {correct}",
        f"- Partial: {partial}",
        f"- Errors: {errors}",
        f"- Passed: {passed}",
        f"- Pass rate: {(passed / total * 100) if total else 0:.1f}%",
        f"- Average score: {avg_score:.3f}",
        f"- Average judge latency: {avg_latency:.0f} ms",
        "",
        "## Case Results",
        "",
        "| ID | Verdict | Score | Passed | Latency ms | Provider | Model |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['verdict']} | {row['score']:.3f} | {row['passed']} | "
            f"{row['latency_ms']} | {row['provider']} | {row['model']} |"
        )
    lines.extend(
        [
            "",
            "## Common Missing Facts",
            "",
            *_missing_lines(rows),
            "",
            "## Slowest Cases",
            "",
            *_slow_lines(rows),
        ]
    )
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _missing_lines(rows: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for item in row.get("missing", []):
            counts[item] = counts.get(item, 0) + 1
    if not counts:
        return ["- None"]
    return [f"- {item}: {count}" for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]]


def _slow_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- None"]
    return [
        f"- {row['id']}: {row['latency_ms']} ms ({row['provider']}/{row['model']})"
        for row in sorted(rows, key=lambda item: int(item["latency_ms"]), reverse=True)[:10]
    ]
