# Examples

## Quick Smoke Test

```bash
llm-judge evaluate \
  --input examples/rag_eval.jsonl \
  --profile default \
  --mode quick \
  --out .llm-judge-runs/default-quick
```

## Replay Audit

```bash
llm-judge evaluate \
  --input examples/rag_eval.jsonl \
  --limit 1 \
  --mode quick \
  --audit \
  --out .llm-judge-runs/audit-smoke
```

Inspect `audit/<case-id>/prompt-judge.txt`, `case.json`, `chunks.json`, and `raw.json` to replay or debug what happened.

## RAGAS Output

```bash
llm-judge evaluate \
  --input examples/ragas_eval.jsonl \
  --profile ragas \
  --mode quick \
  --out .llm-judge-runs/ragas
```

## Chunkshop E1-E8

```bash
llm-judge evaluate \
  --input examples/chunkshop_e1e8.jsonl \
  --profile chunkshop-e1e8 \
  --generate-answer \
  --answer-provider openai-compatible \
  --answer-model gpt-4.1-mini \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --cache-dir .llm-judge-cache \
  --resume \
  --out .llm-judge-runs/chunkshop
```

## Missing Gold/Reference Answers

```bash
llm-judge evaluate \
  --input examples/baseline_reference_generation.jsonl \
  --generate-expected \
  --expected-provider openai-compatible \
  --expected-model gpt-4.1-mini \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/baseline-reference
```

The broad birthplace case can give full credit to semantically true variants such as state, city, or site. The specific city-and-hospital case should require both requested fields, so city-only is partial.

## Local Two-Judge Setup

```bash
llm-judge evaluate --config examples/local_two_judges.yaml
```

## Local vLLM/OpenAI-Compatible No-Key Setup

```bash
llm-judge evaluate --config examples/local_vllm_no_key.yaml
```

This example omits `api_key_env` and disables provider-native `response_format`, which is often useful for local OpenAI-compatible servers that do not implement strict JSON mode.

## Three-Judge Setup

```bash
llm-judge evaluate --config examples/three_judges.yaml
```
