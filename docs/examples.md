# Examples

## Quick Smoke Test

```bash
llm-judge evaluate \
  --input examples/rag_eval.jsonl \
  --profile default \
  --mode quick \
  --out .llm-judge-runs/default-quick
```

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

## Local Two-Judge Setup

```bash
llm-judge evaluate --config examples/local_two_judges.yaml
```

## Three-Judge Setup

```bash
llm-judge evaluate --config examples/three_judges.yaml
```
