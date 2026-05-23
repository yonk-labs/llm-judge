# CLI Reference

## Commands

### `profiles`

Prints supported input profiles and their mapped fields.

```bash
python3 -m llm_judge profiles
```

### `evaluate`

Loads cases, optionally generates missing answers, judges each case, and writes audit output.

```bash
python3 -m llm_judge evaluate --input cases.jsonl --mode quick
```

## `evaluate` Parameters

| Parameter | Type | Default | Description |
|---|---:|---|---|
| `--config` | path | none | YAML run configuration. Can include input, output, answer provider, and up to three judges. |
| `--input` | path | required | JSONL, JSON, YAML, or CSV input file. |
| `--profile` | enum | `default` | Input schema profile. Run `profiles` for choices. |
| `--out` | path | `.llm-judge-runs/<timestamp>` | Output directory. |
| `--mode` | enum | `quick` | `quick`, `accurate`, or `dual`. |
| `--synonyms` | path | none | JSON synonym map for quick scoring. |
| `--provider` | string | `openai-compatible` | Judge provider. |
| `--model` | string | provider default | Judge model. |
| `--base-url` | URL | provider default | Judge provider base URL. |
| `--api-key-env` | string | provider default | Environment variable containing judge API key. |
| `--judge-command` | string | none | Command provider executable for judge calls. |
| `--generate-answer` | flag | false | Generate answers when input answer is empty. |
| `--answer-provider` | string | `--provider` | Answer-generation provider. |
| `--answer-model` | string | `--model` | Answer-generation model. |
| `--answer-base-url` | URL | `--base-url` | Answer provider base URL. |
| `--answer-api-key-env` | string | `--api-key-env` | Answer provider API key env var. |
| `--answer-command` | string | `--judge-command` | Command provider executable for answer calls. |
| `--concurrency` | int | `1` | Number of case worker threads. |
| `--cache-dir` | path | none | Persistent prompt-hash cache directory. |
| `--resume` | flag | false | Skip case IDs already present in existing `results.jsonl`. |
| `--parse-retries` | int | `1` | Judge JSON repair retry count. |
| `--retries` | int | `2` | Provider HTTP retry count. |
| `--timeout` | float | `120.0` | Provider HTTP timeout in seconds. |
| `--temperature` | float | `0.0` | Provider sampling temperature where supported. |
| `--llm-threshold` | float | `0.72` | In `dual` mode, quick scores below this go to the LLM judge. |

## Provider Names

- `mock`
- `command`
- `openai-compatible`
- `openai`
- `openrouter`
- `ollama`
- `anthropic`
- `gemini`

## Environment Variables

Default API key env vars:

- OpenAI/OpenAI-compatible: `OPENAI_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY`

Use `--api-key-env` or `--answer-api-key-env` to override.

API keys are accepted by environment variable name only. There is intentionally no `--api-key` flag, and YAML configs should not contain literal secret values. Set the key in your shell or CI secret store, then point `llm-judge` at that variable:

```bash
export OPENAI_API_KEY=...
llm-judge evaluate \
  --input cases.jsonl \
  --mode accurate \
  --provider openai-compatible \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini \
  --api-key-env OPENAI_API_KEY
```

Local OpenAI-compatible endpoints can omit `--api-key-env` when the endpoint does not require authentication. Hosted OpenAI/OpenRouter/Anthropic/Gemini providers require a key in the configured env var.

## YAML Configuration

`--config` values act as defaults. Explicit CLI flags override matching top-level YAML values. For a `judges:` list, provider settings come from each list item.

```yaml
input: examples/chunkshop_e1e8.jsonl
profile: chunkshop-e1e8
out: .llm-judge-runs/chunkshop-e1e8
mode: accurate
generate_answer: true
concurrency: 8
cache_dir: .llm-judge-cache
resume: true
timeout: 120
retries: 2
parse_retries: 1

answer:
  provider: openai-compatible
  model: gpt-4.1-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY

judges:
  - provider: openai-compatible
    model: gpt-4.1-mini
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  - provider: ollama
    model: qwen2.5:14b
    base_url: http://localhost:11434
  - provider: anthropic
    model: claude-3-5-sonnet-latest
    api_key_env: ANTHROPIC_API_KEY
```

`judges` supports one to three entries. If all judges fail, the final verdict is `ERROR`. If some judges fail, successful judges are still aggregated.

See `examples/llm_config.sample.yaml` for a copyable OpenAI-compatible/OpenRouter/Ollama template.
