import json

from llm_judge.models import EvalCase
from llm_judge.cli import main
from llm_judge.config import load_run_config
from llm_judge.engine import evaluate_case, evaluate_cases, generate_reference_with_providers, score_with_judges
from llm_judge.io import load_cases
from llm_judge.providers import LLMProvider, LLMProviderError, LLMResponse, OpenAICompatibleProvider, build_provider
from llm_judge.scorers import generate_answer, generate_reference, llm_score
from llm_judge.scorers import quick_score


def test_quick_score_accepts_abbreviated_case_name() -> None:
    case = EvalCase(
        case_id="case",
        question="Which case?",
        answer="The summary relies on Bostock.",
        expected="Bostock v. Clayton County",
        chunks=["Bostock v. Clayton County is discussed."],
    )

    result = quick_score(case)

    assert result.verdict in {"CORRECT", "PARTIAL"}
    assert result.score >= 0.58


def test_quick_score_penalizes_missing_fact() -> None:
    case = EvalCase(
        case_id="rate",
        question="What rate?",
        answer="Retention was high.",
        expected="Retention was 11 out of 16.",
        chunks=["The raw retrieval had 11/16 relevant items retained."],
    )

    result = quick_score(case)

    assert result.score < 0.82


def test_ragas_profile_maps_standard_fields(tmp_path) -> None:
    path = tmp_path / "ragas.jsonl"
    path.write_text(
        '{"user_input":"Q?","response":"A","reference":"A","retrieved_contexts":["ctx"],"run":"x"}\n',
        encoding="utf-8",
    )

    [case] = load_cases(path, profile="ragas")

    assert case.question == "Q?"
    assert case.answer == "A"
    assert case.expected == "A"
    assert case.chunks == ["ctx"]
    assert case.metadata["input_profile"] == "ragas"
    assert case.metadata["run"] == "x"


def test_longbench_profile_maps_answers_list(tmp_path) -> None:
    path = tmp_path / "longbench.jsonl"
    path.write_text(
        '{"input":"Q?","prediction":"A","answers":["A","Alt"],"context":"ctx"}\n',
        encoding="utf-8",
    )

    [case] = load_cases(path, profile="longbench")

    assert case.question == "Q?"
    assert case.answer == "A"
    assert case.expected == "A; Alt"
    assert case.chunks == ["ctx"]


def test_pgraggraph_e2e_profile_maps_nested_cell(tmp_path) -> None:
    path = tmp_path / "pgrg.json"
    path.write_text(
        """{
          "rows": [{
            "cell": {
              "dataset": "mhr",
              "arm": "lede_spacy",
              "rung": "L1_naive",
              "qid": "mhr:q:1",
              "question": "Q?",
              "answers": ["A"],
              "chunks": [{"content_preview": "A appears here", "score": 0.9}]
            },
            "judge_answer": "A",
            "judge_score": 1.0
          }]
        }""",
        encoding="utf-8",
    )

    [case] = load_cases(path, profile="pgraggraph-e2e")

    assert case.case_id == "mhr:q:1"
    assert case.question == "Q?"
    assert case.answer == "A"
    assert case.expected == "A"
    assert case.chunks == ["A appears here"]
    assert case.settings["dataset"] == "mhr"
    assert case.settings["rung"] == "L1_naive"


def test_graphrag_bench_profile_reads_yaml_questions(tmp_path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(
        """
corpus: demo
questions:
  - id: q1
    question: What matters?
    gold_answer: The exact answer.
    required_facts:
      - The exact answer is present.
    question_class: Fact Retrieval
""",
        encoding="utf-8",
    )

    [case] = load_cases(path, profile="graphrag-bench")

    assert case.case_id == "q1"
    assert case.question == "What matters?"
    assert case.expected == "The exact answer."
    assert case.settings["question_class"] == "Fact Retrieval"
    assert case.settings["corpus"] == "demo"


def test_chunkshop_profile_preserves_required_facts(tmp_path) -> None:
    path = tmp_path / "chunkshop.jsonl"
    path.write_text(
        '{"id":"e1","question":"Q?","gold_answer":"A","required_facts":["fact one"],'
        '"retrieved_chunks":["ctx"],"config_label":"E1","token_counts":{"retrieved":10}}\n',
        encoding="utf-8",
    )

    [case] = load_cases(path, profile="chunkshop-e1e8")

    assert case.case_id == "e1"
    assert case.answer == ""
    assert case.expected == "A"
    assert case.expected_facts == ["fact one"]
    assert case.chunks == ["ctx"]
    assert case.settings["config_label"] == "E1"
    assert case.settings["token_counts"] == {"retrieved": 10}


class PlainProvider(LLMProvider):
    name = "plain"
    model = "plain-model"

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        assert not json_mode
        return LLMResponse(text="Generated answer.", latency_ms=12)


class BadJsonProvider(LLMProvider):
    name = "bad-json"
    model = "bad-model"

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        assert json_mode
        return LLMResponse(text="not json", latency_ms=3)


class JsonJudgeProvider(LLMProvider):
    def __init__(self, verdict: str, score: float, model: str) -> None:
        self.name = "json"
        self.model = model
        self.verdict = verdict
        self.score = score

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        assert json_mode
        return LLMResponse(
            text=(
                '{"verdict":"%s","score":%.2f,"answer_score":%.2f,'
                '"retrieval_score":1.0,"supported":["x"],"missing":[],"contradictions":[],'
                '"rationale":"ok"}'
            )
            % (self.verdict, self.score, self.score),
            latency_ms=5,
        )


class ReferenceProvider(LLMProvider):
    name = "reference"
    model = "reference-model"

    def __init__(self, expected: str = "Grand Rapids, Michigan.") -> None:
        self.expected = expected

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        assert json_mode
        return LLMResponse(
            text=(
                '{"expected":"%s","expected_facts":["Answers at the requested granularity."],'
                '"acceptable_answers":["Michigan","Grand Rapids"],'
                '"rationale":"Where-question accepts broader true locations."}'
            )
            % self.expected,
            latency_ms=9,
        )


def test_generate_answer_uses_plain_text_mode() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="", expected="A", chunks=["ctx"])

    result = generate_answer(case, PlainProvider())

    assert result.answer == "Generated answer."
    assert result.error is None


def test_generate_reference_uses_oracle_context() -> None:
    case = EvalCase(
        case_id="birthplace",
        question="Where was Matt born?",
        answer="",
        expected="",
        chunks=["retrieved snippet"],
        reference_contexts=["Matt was born at St Mary's Hospital in Grand Rapids, Michigan."],
    )

    result = generate_reference(case, ReferenceProvider())

    assert result.expected == "Grand Rapids, Michigan."
    assert result.acceptable_answers == ["Michigan", "Grand Rapids"]
    assert result.expected_facts == ["Answers at the requested granularity."]


def test_quick_score_accepts_generated_reference_variants() -> None:
    case = EvalCase(
        case_id="birthplace",
        question="Where was Matt born?",
        answer="Michigan.",
        expected="St Mary's Hospital in Grand Rapids, Michigan.",
        metadata={"acceptable_answers": ["Michigan", "Grand Rapids"]},
    )

    result = quick_score(case)

    assert result.verdict == "CORRECT"
    assert result.raw["expected_variant"] == "Michigan"


def test_quick_score_scores_required_fact_coverage_as_partial() -> None:
    case = EvalCase(
        case_id="birthplace-specific",
        question="What city and hospital was Matt born at?",
        answer="He was born in Grand Rapids.",
        expected="Matt was born at St Mary's Hospital in Grand Rapids.",
        expected_facts=["City: Grand Rapids", "Hospital: St Mary's Hospital"],
        metadata={"acceptable_answers": ["Grand Rapids"]},
    )

    result = quick_score(case)

    assert result.verdict == "PARTIAL"
    assert result.score == 0.5
    assert result.supported == ["City: Grand Rapids"]
    assert result.missing == ["Hospital: St Mary's Hospital"]


def test_quick_score_full_credit_when_all_required_facts_present() -> None:
    case = EvalCase(
        case_id="birthplace-specific",
        question="What city and hospital was Matt born at?",
        answer="He was born at St Mary's Hospital in Grand Rapids.",
        expected="Matt was born at St Mary's Hospital in Grand Rapids.",
        expected_facts=["City: Grand Rapids", "Hospital: St Mary's Hospital"],
    )

    result = quick_score(case)

    assert result.verdict == "CORRECT"
    assert result.score == 1.0
    assert result.missing == []


def test_llm_score_returns_error_on_unparseable_json() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="A", expected="A", chunks=["ctx"])

    result = llm_score(case, BadJsonProvider(), parse_retries=0)

    assert result.verdict == "ERROR"
    assert result.score == 0.0


def test_ensemble_error_preserves_individual_judge_details() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="A", expected="A", chunks=["ctx"])

    decision = score_with_judges(case, [BadJsonProvider(), BadJsonProvider()], parse_retries=0)

    assert decision.verdict == "ERROR"
    assert decision.provider == "ensemble"
    assert len(decision.raw["individual_judges"]) == 2
    assert decision.raw["individual_judges"][0]["verdict"] == "ERROR"
    assert "judge failed" in decision.raw["individual_judges"][0]["rationale"]
    assert "error" in decision.raw["individual_judges"][0]["raw"]


def test_evaluate_cases_generates_missing_answer_and_resumes(tmp_path) -> None:
    case = EvalCase(case_id="x", question="Q?", answer="", expected="Generated answer.", chunks=["ctx"])

    rows = evaluate_cases(
        [case],
        out_dir=tmp_path,
        mode="quick",
        synonyms={},
        answer_provider=PlainProvider(),
        generate_missing_answer=True,
        resume=True,
    )

    assert rows[0]["answer"] == "Generated answer."
    rows2 = evaluate_cases(
        [case],
        out_dir=tmp_path,
        mode="quick",
        synonyms={},
        answer_provider=PlainProvider(),
        generate_missing_answer=True,
        resume=True,
    )
    assert rows2 == rows


def test_cli_limit_only_evaluates_first_n_cases(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"one","question":"Q1?","answer":"A","expected":"A"}',
                '{"id":"two","question":"Q2?","answer":"A","expected":"A"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    status = main(["evaluate", "--input", str(path), "--limit", "1", "--out", str(out_dir)])

    assert status == 0
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["one"]


def test_cli_audit_writes_replay_files(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"audit-one","question":"Q?","answer":"Alpha","expected":"Alpha","chunks":["ctx"],"settings":{"k":1}}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    status = main(["evaluate", "--input", str(path), "--audit", "--out", str(out_dir)])

    assert status == 0
    audit_dir = out_dir / "audit" / "audit-one"
    assert (audit_dir / "case.json").exists()
    assert (audit_dir / "chunks.json").exists()
    assert (audit_dir / "prompts.json").exists()
    assert (audit_dir / "raw.json").exists()
    assert "Question:\nQ?" in (audit_dir / "prompt-judge.txt").read_text(encoding="utf-8")
    case_payload = json.loads((audit_dir / "case.json").read_text(encoding="utf-8"))
    assert case_payload["chunks"] == ["ctx"]
    assert case_payload["source_record"]["settings"] == {"k": 1}
    raw_payload = json.loads((audit_dir / "raw.json").read_text(encoding="utf-8"))
    assert raw_payload["decision"]["verdict"] == "CORRECT"


def test_evaluate_case_can_generate_expected_then_answer() -> None:
    case = EvalCase(
        case_id="birthplace",
        question="Where was Matt born?",
        answer="",
        expected="",
        chunks=["Grand Rapids is in the retrieved chunk."],
        reference_contexts=["Matt was born at St Mary's Hospital in Grand Rapids, Michigan."],
    )

    judged_case, decision = evaluate_case(
        case,
        mode="quick",
        synonyms={},
        answer_provider=PlainProvider(),
        reference_provider=ReferenceProvider(),
        generate_missing_answer=True,
        generate_missing_expected=True,
    )

    assert judged_case.expected == "Grand Rapids, Michigan."
    assert judged_case.answer == "Generated answer."
    assert "acceptable_answers" in judged_case.metadata
    assert decision.verdict in {"CORRECT", "PARTIAL", "INCORRECT"}


def test_generate_reference_with_multiple_providers_keeps_variants() -> None:
    case = EvalCase(case_id="x", question="Where?", answer="", expected="", chunks=["ctx"])

    result = generate_reference_with_providers(
        case,
        [ReferenceProvider("Grand Rapids, Michigan."), ReferenceProvider("St Mary's Hospital.")],
    )

    assert result.provider == "reference-ensemble"
    assert "Grand Rapids, Michigan." in result.acceptable_answers
    assert "St Mary's Hospital." in result.acceptable_answers


def test_score_with_three_judges_uses_majority() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="A", expected="A", chunks=["A"])
    decision = score_with_judges(
        case,
        [
            JsonJudgeProvider("CORRECT", 1.0, "j1"),
            JsonJudgeProvider("WRONG", 0.0, "j2"),
            JsonJudgeProvider("CORRECT", 0.9, "j3"),
        ],
    )

    assert decision.verdict == "CORRECT"
    assert decision.provider == "ensemble"
    assert decision.score == 0.633
    assert len(decision.raw["individual_judges"]) == 3


def test_evaluate_case_accepts_judge_provider_list() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="A", expected="A", chunks=["A"])

    _, decision = evaluate_case(
        case,
        mode="accurate",
        synonyms={},
        judge_providers=[
            JsonJudgeProvider("PARTIAL", 0.6, "j1"),
            JsonJudgeProvider("PARTIAL", 0.7, "j2"),
        ],
    )

    assert decision.verdict == "PARTIAL"
    assert decision.score == 0.65


def test_yaml_config_rejects_more_than_three_judges(tmp_path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        """
input: cases.jsonl
judges:
  - provider: mock
  - provider: mock
  - provider: mock
  - provider: mock
""",
        encoding="utf-8",
    )

    try:
        load_run_config(path)
    except ValueError as exc:
        assert "at most 3 judges" in str(exc)
    else:
        raise AssertionError("expected config validation failure")


def test_yaml_config_rejects_more_than_three_references(tmp_path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        """
input: cases.jsonl
references:
  - provider: mock
  - provider: mock
  - provider: mock
  - provider: mock
""",
        encoding="utf-8",
    )

    try:
        load_run_config(path)
    except ValueError as exc:
        assert "at most 3 references" in str(exc)
    else:
        raise AssertionError("expected config validation failure")


def test_openai_compatible_local_endpoint_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = build_provider(
        provider="openai-compatible",
        model="local-model",
        base_url="http://192.168.1.193:8000/v1",
        api_key_env=None,
        command=None,
        timeout=1.0,
        temperature=0.0,
    )

    assert provider.model == "local-model"


def test_openai_provider_can_disable_response_format_and_set_max_tokens() -> None:
    provider = OpenAICompatibleProvider(
        name="openai-compatible",
        model="local-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key=None,
        max_tokens=77,
        disable_response_format=True,
    )
    captured = {}

    def fake_post(url, body, headers):
        captured["body"] = body
        return LLMResponse(text="{}", latency_ms=1)

    provider._post = fake_post  # type: ignore[method-assign]

    provider.complete("prompt", json_mode=True)

    assert captured["body"]["max_tokens"] == 77
    assert "response_format" not in captured["body"]


def test_openai_provider_strict_json_fallback_retries_without_response_format() -> None:
    provider = OpenAICompatibleProvider(
        name="openai-compatible",
        model="local-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key=None,
        strict_json_fallback=True,
    )
    bodies = []

    def fake_post(url, body, headers):
        bodies.append(body)
        if "response_format" in body:
            raise LLMProviderError("unsupported response_format")
        return LLMResponse(text="{}", latency_ms=1)

    provider._post = fake_post  # type: ignore[method-assign]

    provider.complete("prompt", json_mode=True)

    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]
