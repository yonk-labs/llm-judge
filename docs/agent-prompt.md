# Agent Prompt

Use this prompt with Codex, Claude Code, or another coding agent when you want the agent to install `llm-judge` into a benchmark project and use it consistently.

```text
Install and use the public llm-judge utility from https://github.com/yonk-labs/llm-judge for this project’s benchmark evaluation.

Goals:
- Evaluate RAG/chunking benchmark accuracy with inspectable per-case audits.
- Preserve each question, retrieved chunks/context, settings/config labels, generated answer, expected answer, required facts, judge verdict, rationale, and timing.
- Use quick mode for smoke tests and accurate mode for final/disputed runs.
- If benchmark records have retrieved chunks but no answer, use llm-judge answer generation before judging.
- Use --cache-dir and --resume for long sweeps.
- Do not store API keys in files. Use environment variables only.

Steps:
1. Install:
   python3 -m pip install "llm-judge @ git+https://github.com/yonk-labs/llm-judge.git"
2. Inspect the benchmark output schema and choose or adapt an input profile:
   default, ragas, locomo, longbench, langbench, pgraggraph-e2e, benchmark-results, graphrag-bench, hotpotqa, musique, twowiki, multihop-rag, chunkshop-e1e8.
3. Create a JSONL/JSON/YAML/CSV input file if the benchmark does not already produce one.
4. Create a YAML config using up to three judges when useful:
   - one answer provider if answers need to be generated
   - one to three judge providers
   - cache_dir, resume, concurrency, retries, parse_retries
5. Run:
   llm-judge evaluate --config path/to/run.yaml
6. Review:
   - summary.md
   - results.jsonl
   - cases/*.md
7. Report aggregate accuracy and include concrete examples where the judge disagreed with old exact-match/substr scoring.

Important:
- Provider/runtime failures should produce ERROR rows, not abort the run.
- Do not rely on exact substring scoring as the final answer-quality metric.
- Treat abbreviations, aliases, synonyms, reordered facts, and paraphrases as potentially correct when unambiguous.
- Preserve all audit output in an ignored folder such as .llm-judge-runs/.
```
