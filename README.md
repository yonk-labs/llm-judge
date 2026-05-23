# llm-judge

Portable CLI for judging RAG and LLM benchmark runs across local, OpenAI-compatible, and cloud LLM providers.

## Documentation

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [User Guide](docs/user-guide.md)
- [CLI Reference](docs/cli-reference.md)
- [API Guide](docs/api-guide.md)
- [Input Profiles](docs/profiles.md)
- [Operations Notes](docs/operations.md)
- [Examples](docs/examples.md)
- [Agent Prompt](docs/agent-prompt.md)

The main design goal is fair, inspectable evaluation:

- Quick mode: deterministic heuristic scorer for cheap smoke tests.
- Accurate mode: LLM-as-judge with a paraphrase-tolerant rubric and strict JSON output.
- Dual mode: runs quick first, then LLM judge when configured.
- Answer generation: optional retrieved-context -> answer step before judging.
- Per-case audit files: question, chunks, settings, answer, expected answer, judge decision, timing.
- Long-run behavior: retries, non-fatal `ERROR` verdicts, concurrency, resume, and prompt cache.
- Provider support: OpenAI-compatible endpoints, OpenAI, OpenRouter, Ollama, Anthropic, Gemini, shell command, and mock.

## Input Format

Use JSONL, JSON, or CSV. Each JSONL line is one case:

```json
{"id":"q1","question":"What case did the summary mention?","answer":"It cites Bostock.","expected":"Bostock v. Clayton County","chunks":["The decision in Bostock v. Clayton County is discussed."],"settings":{"retriever":"hybrid","k":8}}
```

Recognized aliases:

- Expected answer: `expected`, `gold`, `gold_answer`, `reference`, `reference_answer`
- Required facts: `required_facts`, `expected_facts`
- Answer: `answer`, `actual`, `output`, `llm_answer`, `response`
- Chunks: `chunks`, `contexts`, `retrieved_chunks`, `retrieved_contexts`
- Full/oracle context for reference generation: `reference_context`, `reference_contexts`, `oracle_context`, `full_context`, `full_data`

## Quick Start

Install:

```bash
python3 -m pip install "llm-judge @ git+https://github.com/yonk-labs/llm-judge.git"
```

Run:

```bash
python3 -m llm_judge evaluate \
  --input examples/rag_eval.jsonl \
  --mode quick \
  --out .llm-judge-runs/demo
```

## Standard Benchmark Profiles

Use `--profile` when importing benchmark outputs that already have their own field names:

```bash
python3 -m llm_judge profiles
```

Supported profiles:

- `default`: project-local JSONL with flexible aliases.
- `ragas`: maps `user_input`, `response`, `reference`, `retrieved_contexts`/`contexts`.
- `locomo`: maps long conversational memory outputs with `question`, `response`/`prediction`, `answer`/`reference`, `conversation`/`context`.
- `longbench`: maps `input`, `prediction`, `answers`, `context`.
- `langbench`: alias profile for LangBench/LongBench-like outputs using `input`, `prediction`, `answers`, `context`.
- `pgraggraph-e2e`: maps `../pg-raggraph/benchmarks/e2e/results/*.json` nested `cell.*` rows.
- `benchmark-results`: maps common result rows with `question`, generated `answer`, and `gold`/`reference`.
- `graphrag-bench`: maps AGE bakeoff and GraphRAG-Bench YAML questions with `gold_answer`/`required_facts`.
- `hotpotqa`, `musique`, `twowiki`, `multihop-rag`: maps common third-party QA benchmark records.
- `chunkshop-e1e8`: maps Chunkshop E1-E8 records with `gold_answer`, `required_facts`, retrieved/summarized contexts, retrievable flags, token counts, and config labels.

RAGAS-style output:

```bash
python3 -m llm_judge evaluate \
  --input ragas-results.jsonl \
  --profile ragas \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/ragas
```

LoCoMo-style output:

```bash
python3 -m llm_judge evaluate \
  --input locomo-results.jsonl \
  --profile locomo \
  --mode dual \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/locomo
```

LongBench/LangBench-style output:

```bash
python3 -m llm_judge evaluate \
  --input longbench-predictions.jsonl \
  --profile longbench \
  --mode quick \
  --out .llm-judge-runs/longbench
```

pg-raggraph e2e output:

```bash
python3 -m llm_judge evaluate \
  --input ../pg-raggraph/benchmarks/e2e/results/2026-05-20-mhr-lede_spacy.json \
  --profile pgraggraph-e2e \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/pg-raggraph-mhr
```

MuSiQue/HotpotQA-style benchmark result rows:

```bash
python3 -m llm_judge evaluate \
  --input ../pg-raggraph/benchmarks/musique/_results/results-20260429-124617.json \
  --profile benchmark-results \
  --mode dual \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/musique
```

Chunkshop E1-E8 with answer generation:

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

Baseline/reference generation when no gold answer exists:

```bash
python3 -m llm_judge evaluate \
  --input cases-with-full-context.jsonl \
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

Use this when the benchmark has a question and full/oracle source data but no trusted `gold_answer`. The reference generator writes a concise expected answer, strictly required facts, and acceptable answer variants. For broad questions like "Where was Matt born?", variants such as "Michigan", "Grand Rapids", or a specific hospital can be accepted when the source supports them. For specific questions like "What city and hospital was he born at?", the generated required facts should require both fields.

YAML setup with up to three LLM judges:

```bash
python3 -m llm_judge evaluate --config examples/three_judges.yaml
```

Use [examples/llm_config.sample.yaml](examples/llm_config.sample.yaml) as the copyable real-provider template.
Use [examples/local_two_judges.yaml](examples/local_two_judges.yaml) for the local `192.168.1.193:8000` / `192.168.1.133:8000` two-judge setup.

The YAML config can define one answer model, up to three reference/gold generators, and up to three judges. Multiple judges are aggregated into one final decision while preserving each individual judge result in `raw.individual_judges`.

## API Keys

`llm-judge` accepts API keys through environment variables. In CLI mode, pass the variable name with `--api-key-env` for judges and `--answer-api-key-env` for answer generation. In YAML, use `api_key_env`.

There is intentionally no `--api-key` flag and no YAML field for a literal key value. This keeps secrets out of shell history, config files, benchmark artifacts, and git. Local OpenAI-compatible endpoints, such as private LAN vLLM/llama.cpp servers, can omit `api_key_env` when they do not require authentication.

Accurate judge using any OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
python3 -m llm_judge evaluate \
  --input examples/rag_eval.jsonl \
  --mode accurate \
  --provider openai-compatible \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/accurate
```

Ollama:

```bash
python3 -m llm_judge evaluate \
  --input examples/rag_eval.jsonl \
  --mode accurate \
  --provider ollama \
  --base-url http://localhost:11434 \
  --model qwen2.5:14b \
  --out .llm-judge-runs/ollama
```

## Output

Each run writes:

- `summary.md` with aggregate metrics and headers.
- `results.jsonl` with machine-readable per-case results.
- `cases/<id>.md` with the full audit for each case.

The output directory is intentionally ignored by git.

Verdicts are normalized internally. `WRONG`, `FAIL`, and `FAILED` map to `INCORRECT`; provider/runtime failures are reported as `ERROR` rows instead of aborting the whole run.

## Codex Skill

This repo includes an installable agent skill at [skills/llm-judge/SKILL.md](skills/llm-judge/SKILL.md).

Codex:

```bash
mkdir -p ~/.codex/skills/llm-judge
curl -L https://raw.githubusercontent.com/yonk-labs/llm-judge/main/skills/llm-judge/SKILL.md \
  -o ~/.codex/skills/llm-judge/SKILL.md
```

Claude:

```bash
mkdir -p ~/.claude/skills/llm-judge
curl -L https://raw.githubusercontent.com/yonk-labs/llm-judge/main/skills/llm-judge/SKILL.md \
  -o ~/.claude/skills/llm-judge/SKILL.md
```
