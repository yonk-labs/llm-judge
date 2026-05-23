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

## Requirements

- Python 3.10+
- PyYAML
- Optional provider API keys depending on configured providers
- Optional local OpenAI-compatible servers, Ollama, or command providers
