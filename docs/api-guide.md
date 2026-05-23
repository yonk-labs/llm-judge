# API Guide

## Data Models

### `EvalCase`

```python
EvalCase(
    case_id: str,
    question: str,
    answer: str,
    expected: str,
    expected_facts: list[str] = [],
    chunks: list[Any] = [],
    reference_contexts: list[Any] = [],
    settings: dict[str, Any] = {},
    metadata: dict[str, Any] = {},
    source_record: dict[str, Any] = {},
)
```

Use `answer=""` when answer generation should run first.
Use `expected=""` plus `reference_contexts=[...]` when reference/gold generation should run first.

### `AnswerDecision`

Fields: `answer`, `rationale`, `latency_ms`, `provider`, `model`, `raw`, `error`.

### `ReferenceDecision`

Fields: `expected`, `expected_facts`, `acceptable_answers`, `rationale`, `latency_ms`, `provider`, `model`, `raw`, `error`.

### `JudgeDecision`

Fields: `score`, `verdict`, `rationale`, `missing`, `supported`, `contradictions`, `retrieval_score`, `answer_score`, `raw`, `latency_ms`, `provider`, `model`.

`JudgeDecision.passed` returns true for `CORRECT`/`PARTIAL` with score `>= 0.72`.

## Loading Cases

```python
from pathlib import Path
from llm_judge.io import load_cases

cases = load_cases(Path("cases.jsonl"), profile="chunkshop-e1e8")
```

Parameters:

- `path`: JSONL, JSON, YAML, or CSV file.
- `profile`: profile name from `llm_judge.io.PROFILES`.

## Building Providers

```python
from llm_judge.providers import build_provider, CachedProvider

judge = build_provider(
    provider="openai-compatible",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    command=None,
    timeout=120.0,
    temperature=0.0,
    retries=2,
    max_tokens=1200,
    disable_response_format=False,
    strict_json_fallback=True,
)

judge = CachedProvider(judge, Path(".llm-judge-cache"), "judge")
```

`complete(prompt, json_mode=False)` is the provider contract. Set `json_mode=True` only for structured judge calls.

## Generate One Answer

```python
from llm_judge.scorers import generate_answer

answer = generate_answer(case, answer_provider)
if answer.error:
    ...
```

## Generate One Reference

```python
from llm_judge.scorers import generate_reference

reference = generate_reference(case, reference_provider)
if reference.error:
    ...
```

## Judge One Case

```python
from llm_judge.scorers import quick_score, llm_score

quick = quick_score(case, synonyms={})
accurate = llm_score(case, judge_provider, parse_retries=1)
```

`llm_score` returns `JudgeDecision(verdict="ERROR")` on provider or JSON failures.

## Full Case Pipeline

```python
from llm_judge.engine import evaluate_case

judged_case, decision = evaluate_case(
    case,
    mode="dual",
    synonyms={},
    judge_provider=judge,
    reference_provider=reference_provider,
    answer_provider=answer_provider,
    generate_missing_expected=True,
    generate_missing_answer=True,
    llm_threshold=0.72,
    parse_retries=1,
)
```

## Multi-Judge Pipeline

Use `judge_providers` for an ensemble of one to three LLM judges. The aggregate decision uses majority verdict when available and average score as a tie-breaker.

```python
from llm_judge.engine import evaluate_case

judged_case, decision = evaluate_case(
    case,
    mode="accurate",
    synonyms={},
    judge_providers=[judge_a, judge_b, judge_c],
    parse_retries=1,
)

individual = decision.raw["individual_judges"]
```

Use `reference_providers` for one to three reference/gold generators when the benchmark has no gold answer. The first successful reference is used as `expected`; all successful reference answers and variants are preserved as acceptable answers.

## Batch Pipeline

```python
from llm_judge.engine import evaluate_cases

rows = evaluate_cases(
    cases,
    out_dir=Path(".llm-judge-runs/run-001"),
    mode="accurate",
    synonyms={},
    judge_provider=judge,
    # or: judge_providers=[judge_a, judge_b, judge_c],
    reference_provider=reference_provider,
    # or: reference_providers=[ref_a, ref_b, ref_c],
    answer_provider=answer_provider,
    generate_missing_expected=True,
    generate_missing_answer=True,
    concurrency=8,
    llm_threshold=0.72,
    parse_retries=1,
    resume=True,
)
```

`evaluate_cases` writes `results.jsonl` incrementally after each completed case, so interrupted runs can resume.

## Public Function Index

| Module | Function/Class | Purpose |
|---|---|---|
| `llm_judge.io` | `load_cases(path, profile)` | Normalize input records into `EvalCase`. |
| `llm_judge.io` | `dump_jsonl(path, rows)` | Write JSONL rows. |
| `llm_judge.providers` | `build_provider(...)` | Construct a provider from config. |
| `llm_judge.providers` | `CachedProvider(provider, cache_dir, namespace)` | Add prompt-hash cache. |
| `llm_judge.scorers` | `quick_score(case, synonyms)` | Deterministic heuristic judge. |
| `llm_judge.scorers` | `generate_answer(case, provider)` | Generate answer from chunks. |
| `llm_judge.scorers` | `generate_reference(case, provider, parse_retries)` | Generate missing gold/reference answer from full/oracle context. |
| `llm_judge.scorers` | `llm_score(case, provider, parse_retries)` | LLM-as-judge. |
| `llm_judge.engine` | `evaluate_case(...)` | One-case generation/judging pipeline. |
| `llm_judge.engine` | `evaluate_cases(...)` | Batch pipeline with audit output. |
| `llm_judge.engine` | `generate_reference_with_providers(case, providers, parse_retries)` | Aggregate up to three reference/gold generators. |
| `llm_judge.engine` | `score_with_judges(case, providers, parse_retries)` | Aggregate up to three judge providers. |
| `llm_judge.report` | `write_case_audit(case_dir, case, decision)` | Write per-case Markdown audit. |
| `llm_judge.report` | `write_trace_audit(audit_dir, case, decision)` | Write replay-oriented JSON and prompt audit files. |
| `llm_judge.report` | `write_summary(out_dir, rows)` | Write aggregate summary. |
