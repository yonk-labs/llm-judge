---
name: llm-judge
description: "Reusable RAG/LLM benchmark judging workflow. Use when evaluating RAG accuracy, chunking strategies, retrieval settings, generated answers, LLM-as-judge reliability, per-case audit traces, timing, fair paraphrase-aware grading, or quick-vs-accurate judge modes across local, OpenAI-compatible, and cloud models."
triggers: "llm judge, judge rag, rag eval, evaluate chunking, benchmark accuracy, fair judge, audit traces, judge wrong, quick judge, accurate judge, retrieval accuracy, faithfulness, answer correctness, context recall, per-question audit"
---

# LLM Judge

Public repo: https://github.com/yonk-labs/llm-judge

Use this skill when a project needs a portable judge for RAG, chunking, retrieval, summarization, or answer-quality benchmarks.

## Install The Tool

```bash
python3 -m pip install "llm-judge @ git+https://github.com/yonk-labs/llm-judge.git"
```

Install this skill into Codex:

```bash
mkdir -p ~/.codex/skills/llm-judge
curl -L https://raw.githubusercontent.com/yonk-labs/llm-judge/main/skills/llm-judge/SKILL.md \
  -o ~/.codex/skills/llm-judge/SKILL.md
```

Install this skill into Claude:

```bash
mkdir -p ~/.claude/skills/llm-judge
curl -L https://raw.githubusercontent.com/yonk-labs/llm-judge/main/skills/llm-judge/SKILL.md \
  -o ~/.claude/skills/llm-judge/SKILL.md
```

## Core Rule

Never rely only on exact substring matching for answer correctness. Grade against the expected answer, retrieved chunks, and question intent. Accept paraphrases, abbreviations, aliases, synonyms, reordered facts, and partial names when the reference is unambiguous. Penalize contradictions and missing required facts.

## API Key Rule

Use environment variables for provider keys. Pass env var names with `--api-key-env` / `--answer-api-key-env` or YAML `api_key_env`. Do not put raw API key values in prompts, configs, trace files, shell history, or committed docs. Local OpenAI-compatible endpoints can omit `api_key_env` when they do not require authentication.

## Workflow

1. Identify the eval artifact:
   - JSONL cases with `question`, `answer`, `expected`, `chunks`, and `settings`.
   - Existing trace/audit files that can be converted to JSONL.
   - Benchmark output from tools such as lede, raggraph, stele, chunkshop, or project-specific runners.
   - Standard benchmark/eval outputs such as RAGAS, LoCoMo, LongBench/LangBench-like prediction files, or similar QA datasets.

2. Choose mode:
   - `quick`: deterministic smoke test, cheap, good for local iteration.
   - `accurate`: LLM-as-judge, best for final reports and disputed results.
   - `dual`: quick first, LLM judge only where quick score is below threshold.

   If the benchmark has no gold/reference answer, generate the reference first from full/oracle data with `--generate-expected` / `generate_expected: true`. This is separate from `--generate-answer`, which generates the candidate answer from retrieved chunks.

3. Choose input profile:
   - `default`: flexible project-local schema.
   - `ragas`: `user_input`, `response`, `reference`, `retrieved_contexts`/`contexts`.
   - `locomo`: `question`, `response`/`prediction`, `answer`/`reference`, `conversation`/`context`.
   - `longbench`: `input`, `prediction`, `answers`, `context`.
   - `langbench`: LongBench-like alias profile.
   - `pgraggraph-e2e`: pg-raggraph e2e result rows with nested `cell.*`.
   - `benchmark-results`: common rows with generated `answer` and `gold`/`reference`.
   - `graphrag-bench`: YAML question sets with `gold_answer` and `required_facts`.
   - `hotpotqa`, `musique`, `twowiki`, `multihop-rag`: third-party QA benchmark schemas.
   - `chunkshop-e1e8`: Chunkshop E1-E8 records with `gold_answer`, `required_facts`, retrieved/summarized contexts, retrievable flags, token counts, and config labels.

4. Run the reusable CLI:

```bash
python3 -m llm_judge evaluate --input path/to/cases.jsonl --profile default --mode quick --out .llm-judge-runs/run-name
```

For answer generation before judging:

```bash
python3 -m llm_judge evaluate \
  --input path/to/chunkshop-e1e8.jsonl \
  --profile chunkshop-e1e8 \
  --generate-answer \
  --answer-provider openai-compatible \
  --answer-model gpt-4.1-mini \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --concurrency 8 \
  --cache-dir .llm-judge-cache \
  --resume \
  --out .llm-judge-runs/chunkshop-e1e8
```

For baseline/reference generation when no gold exists:

```bash
python3 -m llm_judge evaluate \
  --input path/to/cases.jsonl \
  --generate-expected \
  --expected-provider openai-compatible \
  --expected-model gpt-4.1-mini \
  --generate-answer \
  --answer-provider openai-compatible \
  --answer-model gpt-4.1-mini \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/baseline
```

Use `reference_context`, `oracle_context`, `full_context`, or `full_data` for the complete source data. The generated reference must match the question granularity: a broad "where" question can give full credit to true state/city/site variants, while a "city and hospital" question requires both fields and should score city-only as partial credit.

For a YAML setup with up to three judges:

```yaml
input: path/to/chunkshop-e1e8.jsonl
profile: chunkshop-e1e8
mode: accurate
generate_answer: true
cache_dir: .llm-judge-cache
resume: true
answer:
  provider: openai-compatible
  model: gpt-4.1-mini
judges:
  - provider: openai-compatible
    model: gpt-4.1-mini
  - provider: ollama
    model: qwen2.5:14b
  - provider: openrouter
    model: anthropic/claude-3.5-sonnet
```

Run with:

```bash
python3 -m llm_judge evaluate --config run.yaml
```

For an accurate OpenAI-compatible judge:

```bash
python3 -m llm_judge evaluate \
  --input path/to/cases.jsonl \
  --profile ragas \
  --mode accurate \
  --provider openai-compatible \
  --base-url "$OPENAI_BASE_URL" \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/run-name
```

For a local OpenAI-compatible no-key server such as vLLM, llama.cpp, or LM Studio:

```bash
python3 -m llm_judge evaluate \
  --input path/to/cases.jsonl \
  --mode accurate \
  --provider openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-model-name \
  --disable-response-format \
  --max-tokens 1200 \
  --out .llm-judge-runs/local-vllm
```

Use `--strict-json-fallback` / `--no-strict-json-fallback` to control whether JSON-mode calls are retried without provider-native strict JSON knobs when a server rejects them. Use `--limit` for smoke tests before long sweeps.

For Ollama:

```bash
python3 -m llm_judge evaluate \
  --input path/to/cases.jsonl \
  --profile locomo \
  --mode accurate \
  --provider ollama \
  --base-url http://localhost:11434 \
  --model qwen2.5:14b \
  --out .llm-judge-runs/run-name
```

5. Inspect outputs:
   - `summary.md`: aggregate accuracy, pass rate, latency, slowest cases, missing facts.
   - `results.jsonl`: machine-readable results.
   - `cases/*.md`: per-question audit with question, chunks, settings, answer, expected answer, judge rationale, and timing.

Provider/runtime failures must produce `ERROR` rows and continue the run. Do not accept a harness that aborts a long sweep because one provider call failed, rate-limited, or returned malformed JSON.

6. When the judge looks wrong:
   - Read the relevant `cases/<id>.md`.
   - Check whether retrieved chunks contain the required fact.
   - Check whether the answer expresses the fact via a synonym, acronym, shortened case name, or reordered phrase.
   - If the answer is acceptable but quick mode missed it, rerun `accurate` or add a project synonym map with `--synonyms`.
   - If accurate mode is too strict, adjust the rubric in `llm_judge/scorers.py::judge_prompt` and rerun a calibration set.

## Required Audit Standard

Every reported benchmark comparison should preserve:

- Question.
- Retrieved chunks or summaries.
- Retrieval/generation settings.
- Generated answer.
- Expected answer.
- Judge verdict and rationale.
- Missing/supported/contradicted facts.
- Per-call timing.

Use `.llm-judge-runs/` or another ignored folder for large trace outputs.
