from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .models import EvalCase


EXPECTED_KEYS = ("expected", "gold", "gold_answer", "reference", "reference_answer", "answers")
ANSWER_KEYS = ("answer", "actual", "output", "llm_answer", "response", "prediction", "predicted_answer")
CHUNK_KEYS = (
    "chunks",
    "contexts",
    "retrieved_chunks",
    "retrieved_contexts",
    "reference_contexts",
    "context",
    "evidence",
    "conversation",
    "dialogue",
)

PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "default": {
        "id": ("id", "case_id", "_id", "qid"),
        "question": ("question", "query", "input", "user_input", "prompt"),
        "answer": ANSWER_KEYS,
        "expected": EXPECTED_KEYS,
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": CHUNK_KEYS,
        "settings": ("settings", "config"),
    },
    "ragas": {
        "id": ("id", "case_id", "_id", "qid"),
        "question": ("user_input", "question", "query", "input"),
        "answer": ("response", "answer", "actual", "output", "llm_answer", "prediction"),
        "expected": ("reference", "expected", "gold", "gold_answer", "reference_answer", "rubrics"),
        "expected_facts": ("expected_facts", "required_facts", "rubrics"),
        "chunks": ("retrieved_contexts", "contexts", "reference_contexts", "chunks", "context"),
        "settings": ("settings", "config", "experiment"),
    },
    "locomo": {
        "id": ("id", "case_id", "_id", "qid"),
        "question": ("question", "query", "input"),
        "answer": ("response", "prediction", "predicted_answer", "answer", "output"),
        "expected": ("answer", "expected", "gold_answer", "reference", "answers"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("context", "conversation", "dialogue", "evidence", "memory", "memories", "retrieved_contexts"),
        "settings": ("settings", "config", "question_type", "category"),
    },
    "longbench": {
        "id": ("id", "case_id", "_id", "qid"),
        "question": ("input", "question", "query", "prompt"),
        "answer": ("prediction", "response", "answer", "output", "actual"),
        "expected": ("answers", "reference", "expected", "gold", "gold_answer"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("context", "contexts", "retrieved_contexts", "chunks", "evidence"),
        "settings": ("settings", "config", "dataset", "task"),
    },
    "langbench": {
        "id": ("id", "case_id", "_id", "qid"),
        "question": ("input", "question", "query", "prompt"),
        "answer": ("prediction", "response", "answer", "output", "actual"),
        "expected": ("answers", "reference", "expected", "gold", "gold_answer"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("context", "contexts", "retrieved_contexts", "chunks", "evidence"),
        "settings": ("settings", "config", "dataset", "task"),
    },
    "pgraggraph-e2e": {
        "id": ("cell.qid", "qid", "id"),
        "question": ("cell.question", "question"),
        "answer": ("generated_answer", "answer", "response", "prediction", "judge_answer"),
        "expected": ("cell.answers", "answers", "gold", "reference"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("cell.chunks", "chunks", "retrieved_contexts", "contexts"),
        "settings": ("settings", "config", "cell"),
    },
    "benchmark-results": {
        "id": ("qid", "id", "_id", "case_id"),
        "question": ("question", "query", "input", "prompt"),
        "answer": ("answer", "response", "prediction", "generated_answer", "output", "llm_answer"),
        "expected": ("gold", "gold_answer", "reference", "answers", "expected", "gold_aliases", "answer_aliases"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("chunks", "contexts", "retrieved_contexts", "context", "evidence", "supporting_docs"),
        "settings": ("settings", "config", "mode", "dataset", "corpus", "hop_class", "type", "level", "category"),
    },
    "graphrag-bench": {
        "id": ("id", "case_id", "qid"),
        "question": ("question", "query", "input"),
        "answer": ("answer", "response", "prediction", "generated_answer", "llm_answer"),
        "expected": ("gold_answer", "required_facts", "expected_substring", "expected", "reference", "answer"),
        "expected_facts": ("required_facts", "expected_facts"),
        "chunks": ("chunks", "contexts", "retrieved_contexts", "context", "evidence"),
        "settings": ("settings", "config", "question_class", "category", "corpus"),
    },
    "hotpotqa": {
        "id": ("id", "_id", "qid"),
        "question": ("question", "query"),
        "answer": ("prediction", "response", "generated_answer", "output", "llm_answer"),
        "expected": ("answer", "answers", "gold", "reference"),
        "expected_facts": ("expected_facts", "required_facts", "supporting_facts"),
        "chunks": ("context", "supporting_docs", "supporting_facts", "retrieved_contexts", "chunks"),
        "settings": ("type", "level", "settings", "config"),
    },
    "musique": {
        "id": ("id", "_id", "qid"),
        "question": ("question", "question_text", "query"),
        "answer": ("prediction", "response", "generated_answer", "output", "answer_generated"),
        "expected": ("answer", "answer_text", "gold", "gold_aliases", "answer_aliases"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("paragraphs", "context", "retrieved_contexts", "chunks"),
        "settings": ("hop_class", "question_decomposition", "decomposition", "mode", "settings", "config"),
    },
    "twowiki": {
        "id": ("_id", "id", "qid"),
        "question": ("question", "query"),
        "answer": ("prediction", "response", "generated_answer", "output", "llm_answer"),
        "expected": ("answer", "answers", "aliases", "gold", "reference"),
        "expected_facts": ("expected_facts", "required_facts", "supporting_facts"),
        "chunks": ("context", "supporting_facts", "retrieved_contexts", "chunks"),
        "settings": ("type", "level", "settings", "config"),
    },
    "multihop-rag": {
        "id": ("id", "_id", "qid"),
        "question": ("query", "question", "input"),
        "answer": ("prediction", "response", "generated_answer", "output", "llm_answer"),
        "expected": ("answer", "answers", "gold", "reference"),
        "expected_facts": ("expected_facts", "required_facts"),
        "chunks": ("evidence_list", "evidence", "context", "retrieved_contexts", "chunks"),
        "settings": ("question_type", "type", "settings", "config"),
    },
    "chunkshop-e1e8": {
        "id": ("id", "case_id", "qid", "question_id"),
        "question": ("question", "query", "input"),
        "answer": ("answer", "generated_answer", "llm_answer", "response", "prediction"),
        "expected": ("gold_answer", "expected", "reference", "answer_key"),
        "expected_facts": ("required_facts", "expected_facts", "facts"),
        "chunks": (
            "retrieved_full_context",
            "retrieved_context",
            "retrieved_chunks",
            "chunks",
            "summarized_answer_context",
            "summary_context",
            "context",
        ),
        "settings": (
            "settings",
            "config",
            "config_label",
            "experiment",
            "strategy",
            "retrievable",
            "token_counts",
            "retrieval_mode",
            "chunker",
        ),
    },
}


def _lookup(record: dict[str, Any], key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _first(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = _lookup(record, key)
        if value is not None:
            return value
    return default


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _as_chunks(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [_chunk_to_text(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return [_chunk_to_text(item) for item in parsed] if isinstance(parsed, list) else [_chunk_to_text(parsed)]
            except json.JSONDecodeError:
                pass
        return [value]
    return [_chunk_to_text(value)]


def _as_text_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [_as_text(item) for item in parsed if _as_text(item)]
            except json.JSONDecodeError:
                pass
        return [value]
    return [_as_text(value)]


def _chunk_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        title = _as_text(value.get("title") or value.get("document_source") or value.get("source") or "")
        text = _as_text(
            value.get("content")
            or value.get("content_preview")
            or value.get("paragraph_text")
            or value.get("text")
            or value.get("body")
            or value.get("sentences")
            or value.get("chunk")
        )
        if title and text:
            return f"{title}\n\n{text}"
        return text or json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and isinstance(value[1], list):
            title = _as_text(value[0])
            text = " ".join(_as_text(item) for item in value[1])
            return f"{title}\n\n{text}" if title else text
        return " ".join(_as_text(item) for item in value)
    return str(value)


def _records_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    records.append(json.loads(stripped))
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data]
        for key in ("rows", "questions", "data", "examples", "results"):
            if isinstance(data, dict) and isinstance(data.get(key), list):
                rows = [dict(item) for item in data[key] if isinstance(item, dict)]
                corpus = data.get("corpus") or data.get("dataset") or data.get("namespace")
                if corpus:
                    for row in rows:
                        row.setdefault("corpus", corpus)
                return rows
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [dict(item) for item in data["data"]]
        if isinstance(data, dict) and isinstance(data.get("examples"), list):
            return [dict(item) for item in data["examples"]]
        return [dict(data)]
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML input requires PyYAML to be installed") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data]
        for key in ("rows", "questions", "data", "examples", "results"):
            if isinstance(data, dict) and isinstance(data.get(key), list):
                rows = [dict(item) for item in data[key] if isinstance(item, dict)]
                corpus = data.get("corpus") or data.get("dataset") or data.get("namespace")
                if corpus:
                    for row in rows:
                        row.setdefault("corpus", corpus)
                return rows
        if isinstance(data, dict):
            return [dict(data)]
        return []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported input extension {path.suffix}. Use .jsonl, .json, .yaml, .yml, or .csv.")


def load_cases(path: Path, profile: str = "default") -> list[EvalCase]:
    """Load JSONL, JSON, YAML, or CSV records into normalized ``EvalCase`` objects.

    Parameters
    ----------
    path:
        Input file path.
    profile:
        Schema profile name from ``PROFILES``.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Choose one of: {', '.join(sorted(PROFILES))}")
    mapping = PROFILES[profile]
    cases: list[EvalCase] = []
    records = _records_from_path(path)
    mapped_keys = {
        "id",
        "case_id",
        "settings",
        "config",
        *mapping["question"],
        *mapping["answer"],
        *mapping["expected"],
        *mapping.get("expected_facts", ()),
        *mapping["chunks"],
        *mapping.get("settings", ()),
    }
    for line_no, record in enumerate(records, 1):
        case_id = _as_text(_first(record, mapping.get("id", ("id", "case_id")), f"case-{line_no:04d}"))
        question = _as_text(_first(record, mapping["question"]))
        answer = _as_text(_first(record, mapping["answer"]))
        expected = _as_text(_first(record, mapping["expected"]))
        expected_facts = _as_text_list(_first(record, mapping.get("expected_facts", ()), []))
        chunks = _as_chunks(_first(record, mapping["chunks"], []))
        settings = _settings_from_record(record, mapping.get("settings", ("settings", "config")))
        if isinstance(settings, str):
            try:
                settings = json.loads(settings)
            except json.JSONDecodeError:
                settings = {"raw_settings": settings}
        metadata = {key: value for key, value in record.items() if key not in mapped_keys}
        metadata["input_profile"] = profile
        cases.append(
            EvalCase(
                case_id=case_id,
                question=question,
                answer=answer,
                expected=expected,
                expected_facts=expected_facts,
                chunks=list(chunks),
                settings=dict(settings),
                metadata=metadata,
            )
        )
    return cases


def _settings_from_record(record: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key in keys:
        value = _lookup(record, key)
        if value is None:
            continue
        if isinstance(value, dict) and key in {"settings", "config"}:
            settings.update(value)
        elif isinstance(value, dict) and key == "cell":
            for cell_key in ("dataset", "arm", "namespace", "rung", "strata", "n_candidates", "latency_ms", "git_sha"):
                if cell_key in value:
                    settings[cell_key] = value[cell_key]
        else:
            settings[key.replace(".", "_")] = value
    return settings


def load_jsonl(path: Path) -> list[EvalCase]:
    """Backward-compatible loader for default-profile JSONL/JSON/CSV inputs."""
    return load_cases(path, profile="default")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL with stable key ordering."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
