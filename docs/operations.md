# Operations Notes

## Reliability

Provider failures are converted to `ERROR` decisions. The run continues and the error is preserved in `results.jsonl` and the per-case audit.

For multi-judge YAML setups, each judge is isolated. One failing judge does not fail the case unless all judges fail.

Retry behavior:

- `--retries` controls HTTP/provider retries.
- `--parse-retries` controls malformed judge JSON repair attempts.
- HTTP 408, 409, 429, 500, 502, 503, and 504 are retried.
- `Retry-After` is honored when present and numeric.

## Caching

`--cache-dir` wraps providers with a prompt-hash cache. Cache keys include:

- provider name
- model
- JSON mode flag
- full prompt text

Answer and judge calls use separate cache namespaces.

## Resume

`--resume` reads existing `results.jsonl` in the output directory and skips matching case IDs. Case IDs must be stable across runs.

## Concurrency

`--concurrency` uses a thread pool. This is appropriate for network-bound provider calls. If a provider has strict rate limits, lower concurrency or use a local queue/proxy.

## Source Control

Large run outputs should stay out of git:

- `.llm-judge-runs/`
- `.llm-judge-cache/`

The repo `.gitignore` already ignores `.llm-judge-runs/`; add project-specific cache/output folders as needed.
