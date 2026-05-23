# Installation

## CLI/Library

Install from the public repository:

```bash
python3 -m pip install "llm-judge @ git+https://github.com/yonk-labs/llm-judge.git"
```

Editable install:

```bash
git clone https://github.com/yonk-labs/llm-judge.git
cd llm-judge
python3 -m pip install -e .
```

Run tests:

```bash
python3 -m pytest -q
```

## Codex Skill

Install the bundled skill:

```bash
mkdir -p ~/.codex/skills/llm-judge
curl -L https://raw.githubusercontent.com/yonk-labs/llm-judge/main/skills/llm-judge/SKILL.md \
  -o ~/.codex/skills/llm-judge/SKILL.md
```

After installation, use it in a Codex session with:

```text
$llm-judge evaluate this benchmark run and write audits
```

## Claude Skill

Install the same bundled skill for Claude:

```bash
mkdir -p ~/.claude/skills/llm-judge
curl -L https://raw.githubusercontent.com/yonk-labs/llm-judge/main/skills/llm-judge/SKILL.md \
  -o ~/.claude/skills/llm-judge/SKILL.md
```

## API Keys

The tool accepts API keys by environment variable name:

- CLI: `--api-key-env OPENAI_API_KEY` and `--answer-api-key-env OPENAI_API_KEY`
- YAML: `api_key_env: OPENAI_API_KEY`

Raw key values are intentionally not accepted in CLI flags or YAML. Store keys in your shell, CI secret store, or process manager environment. Local OpenAI-compatible endpoints can omit `api_key_env` when the server does not require authentication.

## Requirements

- Python 3.10+
- PyYAML
- Optional provider API keys depending on configured providers
- Optional local OpenAI-compatible servers, Ollama, or command providers
