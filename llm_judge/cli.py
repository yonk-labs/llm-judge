from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import judge_sections, load_run_config, pick, pick_provider_value, section
from .engine import evaluate_cases
from .io import PROFILES, load_cases
from .providers import CachedProvider, LLMProvider, build_provider
from .scorers import load_synonyms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-judge", description="Portable RAG/LLM judge CLI.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparsers.add_parser("profiles", help="List supported input profiles.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate JSONL cases.")
    evaluate.add_argument("--config", type=Path, help="YAML run configuration with input, providers, and up to 3 judges.")
    evaluate.add_argument("--input", type=Path, help="JSONL, JSON, YAML, or CSV file of evaluation cases.")
    evaluate.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help="Input schema profile for standard benchmark outputs.",
    )
    evaluate.add_argument("--out", type=Path, help="Output directory. Defaults to .llm-judge-runs/<timestamp>.")
    evaluate.add_argument("--mode", choices=["quick", "accurate", "dual"], default=None)
    evaluate.add_argument("--synonyms", help="Optional JSON synonym map for quick scoring.")
    evaluate.add_argument("--provider", help="mock, openai-compatible, openai, openrouter, ollama, anthropic, gemini, command.")
    evaluate.add_argument("--model", help="Judge model name.")
    evaluate.add_argument("--base-url", help="Provider base URL.")
    evaluate.add_argument("--api-key-env", help="Environment variable containing provider API key.")
    evaluate.add_argument("--judge-command", help="Command provider executable. Prompt is passed on stdin.")
    evaluate.add_argument("--generate-answer", action="store_true", default=None, help="Generate missing answers from retrieved chunks before judging.")
    evaluate.add_argument("--answer-provider", help="Answer model provider. Defaults to --provider.")
    evaluate.add_argument("--answer-model", help="Answer model name. Defaults to --model.")
    evaluate.add_argument("--answer-base-url", help="Answer provider base URL. Defaults to --base-url.")
    evaluate.add_argument("--answer-api-key-env", help="Answer provider API key env var. Defaults to --api-key-env.")
    evaluate.add_argument("--answer-command", help="Command provider executable for answer generation.")
    evaluate.add_argument("--concurrency", type=int, help="Number of cases to process concurrently.")
    evaluate.add_argument("--cache-dir", type=Path, help="Persistent prompt cache directory.")
    evaluate.add_argument("--resume", action="store_true", default=None, help="Resume from existing results.jsonl in --out.")
    evaluate.add_argument("--parse-retries", type=int, help="Judge JSON repair retry count.")
    evaluate.add_argument("--retries", type=int, help="Provider HTTP retry count.")
    evaluate.add_argument("--timeout", type=float)
    evaluate.add_argument("--temperature", type=float)
    evaluate.add_argument("--llm-threshold", type=float, help="Quick score threshold before skipping LLM in dual mode.")
    return parser


def run_evaluate(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    input_path = pick(args.input, config, "input")
    if input_path is None:
        raise SystemExit("--input is required unless supplied by --config")
    input_path = Path(input_path)
    profile = pick(args.profile, config, "profile", "default")
    mode = pick(args.mode, config, "mode", "quick")
    out_value = pick(args.out, config, "out")
    out_dir = Path(out_value) if out_value else Path(".llm-judge-runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(input_path, profile=profile)
    synonyms = load_synonyms(pick(args.synonyms, config, "synonyms"))
    timeout = float(pick(args.timeout, config, "timeout", 120.0))
    temperature = float(pick(args.temperature, config, "temperature", 0.0))
    retries = int(pick(args.retries, config, "retries", 2))
    parse_retries = int(pick(args.parse_retries, config, "parse_retries", 1))
    concurrency = int(pick(args.concurrency, config, "concurrency", 1))
    llm_threshold = float(pick(args.llm_threshold, config, "llm_threshold", 0.72))
    resume = bool(pick(args.resume, config, "resume", False))
    generate_answer_flag = bool(pick(args.generate_answer, config, "generate_answer", False))
    cache_value = pick(args.cache_dir, config, "cache_dir")
    cache_dir = Path(cache_value) if cache_value else None

    provider: LLMProvider | None = None
    judge_providers: list[LLMProvider] | None = None
    if mode in {"accurate", "dual"}:
        judge_cfgs = judge_sections(config)
        if judge_cfgs:
            judge_providers = [
                _build_provider_from_config(
                    judge_cfg,
                    fallback_provider="openai-compatible",
                    timeout=timeout,
                    temperature=temperature,
                    retries=retries,
                    cache_dir=cache_dir,
                    cache_namespace=f"judge-{index}",
                )
                for index, judge_cfg in enumerate(judge_cfgs, 1)
            ]
        else:
            provider = build_provider(
                provider=pick(args.provider, config, "provider", "openai-compatible"),
                model=pick(args.model, config, "model"),
                base_url=pick(args.base_url, config, "base_url"),
                api_key_env=pick(args.api_key_env, config, "api_key_env"),
                command=pick(args.judge_command, config, "judge_command"),
                timeout=timeout,
                temperature=temperature,
                retries=retries,
            )
            if cache_dir:
                provider = CachedProvider(provider, cache_dir, "judge")

    answer_provider: LLMProvider | None = None
    if generate_answer_flag:
        answer_cfg = section(config, "answer")
        answer_provider = build_provider(
            provider=pick_provider_value(args.answer_provider, answer_cfg, "provider", pick(args.provider, config, "provider", "openai-compatible")),
            model=pick_provider_value(args.answer_model, answer_cfg, "model", pick(args.model, config, "model")),
            base_url=pick_provider_value(args.answer_base_url, answer_cfg, "base_url", pick(args.base_url, config, "base_url")),
            api_key_env=pick_provider_value(args.answer_api_key_env, answer_cfg, "api_key_env", pick(args.api_key_env, config, "api_key_env")),
            command=pick_provider_value(args.answer_command, answer_cfg, "command", pick(args.judge_command, config, "judge_command")),
            timeout=float(answer_cfg.get("timeout", timeout)),
            temperature=float(answer_cfg.get("temperature", temperature)),
            retries=int(answer_cfg.get("retries", retries)),
        )
        if cache_dir:
            answer_provider = CachedProvider(answer_provider, cache_dir, "answer")

    rows = evaluate_cases(
        cases,
        out_dir=out_dir,
        mode=mode,
        synonyms=synonyms,
        judge_provider=provider,
        judge_providers=judge_providers,
        answer_provider=answer_provider,
        generate_missing_answer=generate_answer_flag,
        concurrency=max(1, concurrency),
        llm_threshold=llm_threshold,
        parse_retries=parse_retries,
        resume=resume,
    )
    summary = out_dir / "summary.md"
    print(f"Wrote {len(rows)} judged cases to {out_dir}")
    print(f"Summary: {summary}")
    return 0


def _build_provider_from_config(
    provider_config: dict,
    *,
    fallback_provider: str,
    timeout: float,
    temperature: float,
    retries: int,
    cache_dir: Path | None,
    cache_namespace: str,
) -> LLMProvider:
    provider = build_provider(
        provider=provider_config.get("provider", fallback_provider),
        model=provider_config.get("model"),
        base_url=provider_config.get("base_url"),
        api_key_env=provider_config.get("api_key_env"),
        command=provider_config.get("command") or provider_config.get("judge_command"),
        timeout=float(provider_config.get("timeout", timeout)),
        temperature=float(provider_config.get("temperature", temperature)),
        retries=int(provider_config.get("retries", retries)),
    )
    if cache_dir:
        return CachedProvider(provider, cache_dir, provider_config.get("cache_namespace", cache_namespace))
    return provider


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand == "profiles":
        for name, mapping in sorted(PROFILES.items()):
            print(f"{name}:")
            print(f"  question: {', '.join(mapping['question'])}")
            print(f"  answer: {', '.join(mapping['answer'])}")
            print(f"  expected: {', '.join(mapping['expected'])}")
            print(f"  chunks: {', '.join(mapping['chunks'])}")
        return 0
    if args.subcommand == "evaluate":
        return run_evaluate(args)
    parser.error(f"unknown command: {args.subcommand}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
