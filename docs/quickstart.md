# Quickstart

This guide gets `llm-judge` running in a benchmark project with local audit output.

## 1. Install

From GitHub:

```bash
python3 -m pip install "llm-judge @ git+https://github.com/yonk-labs/llm-judge.git"
```

For local development:

```bash
git clone https://github.com/yonk-labs/llm-judge.git
cd llm-judge
python3 -m pip install -e .
```

## 2. Prepare Cases

Create `cases.jsonl`:

```json
{"id":"q1","question":"What case did the context mention?","gold_answer":"Bostock v. Clayton County","required_facts":["The answer identifies Bostock v. Clayton County."],"retrieved_chunks":["The decision in Bostock v. Clayton County is discussed."],"config_label":"E1"}
```

## 3. Run Quick Mode

```bash
llm-judge evaluate \
  --input cases.jsonl \
  --profile chunkshop-e1e8 \
  --mode quick \
  --out .llm-judge-runs/quick
```

## 4. Run Accurate Mode

Hosted providers use API keys from environment variables. `llm-judge` intentionally accepts the env var name, not the raw key value, so secrets do not land in YAML, shell history, or benchmark artifacts.

```bash
export OPENAI_API_KEY=...

llm-judge evaluate \
  --input cases.jsonl \
  --profile chunkshop-e1e8 \
  --mode accurate \
  --provider openai-compatible \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini \
  --out .llm-judge-runs/accurate
```

## 5. Generate Missing Answers

If your benchmark has retrieved chunks but no generated answer:

```bash
llm-judge evaluate \
  --input cases.jsonl \
  --profile chunkshop-e1e8 \
  --generate-answer \
  --answer-provider openai-compatible \
  --answer-model gpt-4.1-mini \
  --mode accurate \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --cache-dir .llm-judge-cache \
  --resume \
  --out .llm-judge-runs/generated
```

## 6. Generate Missing Gold Answers

If your benchmark does not have a trusted `gold_answer`, provide full/oracle context in fields such as `reference_context`, `oracle_context`, or `full_context`, then generate the reference before judging:

```bash
llm-judge evaluate \
  --input cases.jsonl \
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

The generated reference includes required facts and acceptable answer variants. The judge uses the question's requested granularity: broad "where" questions can accept state/city/site answers when true, while "city and hospital" questions require both.

## 7. Use YAML

Copy the sample config:

```bash
cp examples/llm_config.sample.yaml run.yaml
llm-judge evaluate --config run.yaml
```

In YAML, use `api_key_env: OPENAI_API_KEY` or another env var name. Do not put the actual key in the YAML file. Local OpenAI-compatible servers that do not require auth can leave `api_key_env` out.

## 8. Inspect Results

Open:

- `.llm-judge-runs/<run>/summary.md`
- `.llm-judge-runs/<run>/results.jsonl`
- `.llm-judge-runs/<run>/cases/<case-id>.md`
