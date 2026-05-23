# User Guide

## Purpose

`llm-judge` is a local-first utility for evaluating RAG and benchmark runs. It can judge existing answers, generate missing answers from retrieved context, generate missing gold/reference answers from full/oracle context, and write inspectable per-case audits.

## Core Workflow

1. Produce cases as JSONL, JSON, YAML, or CSV.
2. Pick an input `--profile`.
3. Pick a mode:
   - `quick`: deterministic heuristic smoke test.
   - `accurate`: LLM-as-judge for final/disputed runs.
   - `dual`: quick first, then LLM judge below `--llm-threshold`.
4. Run the CLI.
5. Inspect `summary.md`, `results.jsonl`, and `cases/*.md`.

## Minimal Case

```json
{"id":"q1","question":"What case?","answer":"Bostock.","expected":"Bostock v. Clayton County","chunks":["Bostock v. Clayton County is discussed."],"settings":{"experiment":"E1"}}
```

## Judging Existing Answers

```bash
python3 -m llm_judge evaluate \
  --input cases.jsonl \
  --profile default \
  --mode accurate \
  --provider openai-compatible \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/run-001
```

## Generating Answers Before Judging

Use this when your benchmark has questions and retrieved chunks but no generated answer yet.

```bash
python3 -m llm_judge evaluate \
  --input chunkshop-e1e8.jsonl \
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

## Generating Gold/Reference Answers

Use this when a benchmark gives you the full source data but no trusted `gold_answer`. Put the full/oracle data in a field such as `reference_context`, `oracle_context`, `full_context`, or `full_data`. Keep retrieved chunks in `chunks`/`retrieved_contexts` so the candidate answer is still generated from the same limited evidence your RAG system saw.

```json
{"id":"birthplace","question":"Where was Matt Yonkovit born?","answer":"He was born in Grand Rapids.","expected":"","reference_context":"Matt Yonkovit was born at St Mary's Hospital in Grand Rapids, Michigan.","chunks":["Matt Yonkovit grew up in Michigan."]}
```

```bash
python3 -m llm_judge evaluate \
  --input cases.jsonl \
  --generate-expected \
  --expected-provider openai-compatible \
  --expected-model gpt-4.1-mini \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/reference-generated
```

You can also generate both sides in one run: `--generate-expected` creates the reference from full/oracle context, then `--generate-answer` creates the candidate answer from retrieved chunks.

For more robust baselines, use up to three reference generators in YAML. The first successful reference becomes the expected answer; all successful expected answers and listed aliases are retained as acceptable answers for semantic judging.

Grading depends on the question granularity. If there are no specific required facts, any supported acceptable answer can be fully correct. For example, "Michigan", "Grand Rapids", and "St Mary's Hospital in Grand Rapids" can each fully answer "Where was Matt Yonkovit born?" if the source supports them. If the question asks "What city and hospital was Matt Yonkovit born at?", use separate required facts for city and hospital; an answer with only the city should be `PARTIAL` with a score around `0.50`.

## YAML Setup With Up To Three Judges

Create a run file:

```yaml
input: examples/chunkshop_e1e8.jsonl
profile: chunkshop-e1e8
out: .llm-judge-runs/three-judges
mode: accurate
generate_answer: true
generate_expected: true
concurrency: 3
cache_dir: .llm-judge-cache
resume: true

answer:
  provider: openai-compatible
  model: gpt-4.1-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY

references:
  - provider: openai-compatible
    model: gpt-4.1-mini
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  - provider: openrouter
    model: anthropic/claude-3.5-sonnet
    api_key_env: OPENROUTER_API_KEY

judges:
  - provider: openai-compatible
    model: gpt-4.1-mini
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  - provider: ollama
    model: qwen2.5:14b
    base_url: http://localhost:11434
  - provider: openrouter
    model: anthropic/claude-3.5-sonnet
    api_key_env: OPENROUTER_API_KEY
```

Run it:

```bash
python3 -m llm_judge evaluate --config run.yaml
```

The final decision aggregates up to three judge verdicts. Individual judge results are preserved in `results.jsonl` under `raw.individual_judges`. Reference-generation details are preserved in each row under `metadata.reference_generation`.

A copyable real-provider template is available at `examples/llm_config.sample.yaml`.

## Long Runs

For sweeps with hundreds or thousands of cases:

- Use `--limit` for smoke tests before running the full benchmark.
- Use `--audit` when debugging a judge or RAG run; it writes replay files with prompts, raw source records, chunks, reference context, normalized case data, and raw provider outputs.
- Use `--cache-dir` to avoid paying twice for identical prompts.
- Use `--resume` so existing case IDs in `results.jsonl` are skipped.
- Use `--concurrency` for parallel calls.
- Keep `--retries` above zero for transient HTTP failures.
- Keep `--parse-retries` above zero for models that sometimes return malformed JSON.
- Use `--disable-response-format` for local OpenAI-compatible providers that reject strict JSON mode.
- Use `--max-tokens` to cap provider output where supported.

Provider failures and malformed judge JSON become `ERROR` rows. They do not abort the run.

When all judges in an ensemble fail, the final row is `ERROR` and preserves each judge's error details under `raw.individual_judges`.

## Output Files

- `summary.md`: aggregate pass rate, mean score, error count, slowest cases, common missing facts.
- `results.jsonl`: one machine-readable row per case.
- `cases/<case-id>.md`: full audit with question, settings, chunks, answer, expected answer, required facts, acceptable answers, supported/missing facts, contradictions, and timing.
- `audit/<case-id>/`: optional `--audit` replay bundle containing `case.json`, `chunks.json`, `prompts.json`, `raw.json`, and plain-text prompt files for answer/reference/judge calls.

## Verdicts

Canonical verdicts:

- `CORRECT`
- `PARTIAL`
- `INCORRECT`
- `ERROR`

Legacy aliases such as `WRONG`, `FAIL`, and `FAILED` normalize to `INCORRECT`.
