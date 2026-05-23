from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import judge_sections, load_run_config, pick, pick_provider_value, reference_sections, section
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
    evaluate.add_argument("--limit", type=int, help="Only evaluate the first N normalized cases.")
    evaluate.add_argument("--mode", choices=["quick", "accurate", "dual"], default=None)
    evaluate.add_argument("--synonyms", help="Optional JSON synonym map for quick scoring.")
    evaluate.add_argument("--provider", help="mock, openai-compatible, openai, openrouter, ollama, anthropic, gemini, command.")
    evaluate.add_argument("--model", help="Judge model name.")
    evaluate.add_argument("--base-url", help="Provider base URL.")
    evaluate.add_argument("--api-key-env", help="Environment variable containing provider API key.")
    evaluate.add_argument("--judge-command", help="Command provider executable. Prompt is passed on stdin.")
    evaluate.add_argument("--generate-expected", "--generate-gold", dest="generate_expected", action="store_true", default=None, help="Generate missing expected/gold answers from full/oracle context before judging.")
    evaluate.add_argument("--expected-provider", "--gold-provider", dest="expected_provider", help="Reference/gold generator provider. Defaults to --answer-provider or --provider.")
    evaluate.add_argument("--expected-model", "--gold-model", dest="expected_model", help="Reference/gold generator model. Defaults to --answer-model or --model.")
    evaluate.add_argument("--expected-base-url", "--gold-base-url", dest="expected_base_url", help="Reference/gold generator base URL.")
    evaluate.add_argument("--expected-api-key-env", "--gold-api-key-env", dest="expected_api_key_env", help="Reference/gold generator API key env var.")
    evaluate.add_argument("--expected-command", "--gold-command", dest="expected_command", help="Command provider executable for reference/gold generation.")
    evaluate.add_argument("--expected-samples", "--gold-samples", dest="expected_samples", type=int, help="Number of reference/gold generations to run with the same CLI provider, max 3.")
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
    evaluate.add_argument("--max-tokens", type=int, help="Provider max output tokens where supported.")
    evaluate.add_argument("--disable-response-format", action="store_true", default=None, help="Do not send provider-native strict JSON response_format/format knobs.")
    evaluate.add_argument("--strict-json-fallback", action=argparse.BooleanOptionalAction, default=None, help="Retry JSON-mode calls without strict response_format when the provider rejects it.")
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
    limit = pick(args.limit, config, "limit")
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise SystemExit("--limit must be >= 0")
        cases = cases[:limit]
    synonyms = load_synonyms(pick(args.synonyms, config, "synonyms"))
    timeout = float(pick(args.timeout, config, "timeout", 120.0))
    temperature = float(pick(args.temperature, config, "temperature", 0.0))
    retries = int(pick(args.retries, config, "retries", 2))
    max_tokens_value = pick(args.max_tokens, config, "max_tokens")
    max_tokens = int(max_tokens_value) if max_tokens_value is not None else None
    disable_response_format = bool(pick(args.disable_response_format, config, "disable_response_format", False))
    strict_json_fallback = bool(pick(args.strict_json_fallback, config, "strict_json_fallback", True))
    parse_retries = int(pick(args.parse_retries, config, "parse_retries", 1))
    concurrency = int(pick(args.concurrency, config, "concurrency", 1))
    llm_threshold = float(pick(args.llm_threshold, config, "llm_threshold", 0.72))
    resume = bool(pick(args.resume, config, "resume", False))
    generate_answer_flag = bool(pick(args.generate_answer, config, "generate_answer", False))
    generate_expected_flag = bool(pick(args.generate_expected, config, "generate_expected", False) or pick(args.generate_expected, config, "generate_gold", False))
    expected_samples = int(pick(args.expected_samples, config, "expected_samples", 1))
    if expected_samples > 3:
        raise SystemExit("--expected-samples supports at most 3")
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
                    max_tokens=max_tokens,
                    disable_response_format=disable_response_format,
                    strict_json_fallback=strict_json_fallback,
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
                max_tokens=max_tokens,
                disable_response_format=disable_response_format,
                strict_json_fallback=strict_json_fallback,
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
            max_tokens=_optional_int(answer_cfg.get("max_tokens", max_tokens)),
            disable_response_format=bool(answer_cfg.get("disable_response_format", disable_response_format)),
            strict_json_fallback=bool(answer_cfg.get("strict_json_fallback", strict_json_fallback)),
        )
        if cache_dir:
            answer_provider = CachedProvider(answer_provider, cache_dir, "answer")

    reference_provider: LLMProvider | None = None
    reference_providers: list[LLMProvider] | None = None
    if generate_expected_flag:
        reference_cfgs = reference_sections(config)
        fallback_answer_cfg = section(config, "answer")
        if not reference_cfgs:
            reference_cfgs = [{} for _ in range(max(1, expected_samples))]
        reference_providers = [
            _build_reference_provider(
                reference_cfg,
                fallback_answer_cfg=fallback_answer_cfg,
                args=args,
                config=config,
                timeout=timeout,
                temperature=temperature,
                retries=retries,
                max_tokens=max_tokens,
                disable_response_format=disable_response_format,
                strict_json_fallback=strict_json_fallback,
                cache_dir=cache_dir,
                cache_namespace=f"reference-{index}",
            )
            for index, reference_cfg in enumerate(reference_cfgs[:3], 1)
        ]
        reference_provider = reference_providers[0] if len(reference_providers) == 1 else None

    rows = evaluate_cases(
        cases,
        out_dir=out_dir,
        mode=mode,
        synonyms=synonyms,
        judge_provider=provider,
        judge_providers=judge_providers,
        answer_provider=answer_provider,
        reference_provider=reference_provider,
        reference_providers=reference_providers,
        generate_missing_answer=generate_answer_flag,
        generate_missing_expected=generate_expected_flag,
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
    max_tokens: int | None,
    disable_response_format: bool,
    strict_json_fallback: bool,
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
        max_tokens=_optional_int(provider_config.get("max_tokens", max_tokens)),
        disable_response_format=bool(provider_config.get("disable_response_format", disable_response_format)),
        strict_json_fallback=bool(provider_config.get("strict_json_fallback", strict_json_fallback)),
    )
    if cache_dir:
        return CachedProvider(provider, cache_dir, provider_config.get("cache_namespace", cache_namespace))
    return provider


def _build_reference_provider(
    provider_config: dict,
    *,
    fallback_answer_cfg: dict,
    args: argparse.Namespace,
    config: dict,
    timeout: float,
    temperature: float,
    retries: int,
    max_tokens: int | None,
    disable_response_format: bool,
    strict_json_fallback: bool,
    cache_dir: Path | None,
    cache_namespace: str,
) -> LLMProvider:
    provider = build_provider(
        provider=pick_provider_value(
            args.expected_provider,
            provider_config,
            "provider",
            pick_provider_value(args.answer_provider, fallback_answer_cfg, "provider", pick(args.provider, config, "provider", "openai-compatible")),
        ),
        model=pick_provider_value(
            args.expected_model,
            provider_config,
            "model",
            pick_provider_value(args.answer_model, fallback_answer_cfg, "model", pick(args.model, config, "model")),
        ),
        base_url=pick_provider_value(
            args.expected_base_url,
            provider_config,
            "base_url",
            pick_provider_value(args.answer_base_url, fallback_answer_cfg, "base_url", pick(args.base_url, config, "base_url")),
        ),
        api_key_env=pick_provider_value(
            args.expected_api_key_env,
            provider_config,
            "api_key_env",
            pick_provider_value(args.answer_api_key_env, fallback_answer_cfg, "api_key_env", pick(args.api_key_env, config, "api_key_env")),
        ),
        command=pick_provider_value(
            args.expected_command,
            provider_config,
            "command",
            pick_provider_value(args.answer_command, fallback_answer_cfg, "command", pick(args.judge_command, config, "judge_command")),
        ),
        timeout=float(provider_config.get("timeout", timeout)),
        temperature=float(provider_config.get("temperature", temperature)),
        retries=int(provider_config.get("retries", retries)),
        max_tokens=_optional_int(provider_config.get("max_tokens", max_tokens)),
        disable_response_format=bool(provider_config.get("disable_response_format", disable_response_format)),
        strict_json_fallback=bool(provider_config.get("strict_json_fallback", strict_json_fallback)),
    )
    if cache_dir:
        return CachedProvider(provider, cache_dir, provider_config.get("cache_namespace", cache_namespace))
    return provider


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


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
