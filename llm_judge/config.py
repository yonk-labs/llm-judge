from __future__ import annotations

from pathlib import Path
from typing import Any


def load_run_config(path: Path | None) -> dict[str, Any]:
    """Load a YAML run configuration.

    Parameters
    ----------
    path:
        YAML file path. ``None`` returns an empty config.
    """
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML to be installed") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML run config must be a mapping")
    judges = data.get("judges")
    if judges is not None:
        if not isinstance(judges, list):
            raise ValueError("YAML 'judges' must be a list")
        if len(judges) > 3:
            raise ValueError("YAML config supports at most 3 judges")
        for index, judge in enumerate(judges, 1):
            if not isinstance(judge, dict):
                raise ValueError(f"YAML judge #{index} must be a mapping")
    return data


def cfg_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return a top-level config value."""
    return config.get(key, default)


def section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a named mapping section or an empty mapping."""
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML '{key}' must be a mapping")
    return value


def judge_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized judge provider sections from ``judges`` or ``judge``."""
    judges = config.get("judges")
    if judges is None:
        single = config.get("judge")
        if single is None:
            return []
        if not isinstance(single, dict):
            raise ValueError("YAML 'judge' must be a mapping")
        return [single]
    if len(judges) > 3:
        raise ValueError("YAML config supports at most 3 judges")
    return judges


def pick(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Choose CLI value first, then YAML value, then a default."""
    return cli_value if cli_value is not None else config.get(key, default)


def pick_provider_value(
    cli_value: Any,
    provider_config: dict[str, Any],
    key: str,
    fallback: Any = None,
) -> Any:
    """Choose a provider-specific CLI/config value."""
    return cli_value if cli_value is not None else provider_config.get(key, fallback)
