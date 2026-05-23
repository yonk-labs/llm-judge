from llm_judge.models import EvalCase
from llm_judge.config import load_run_config
from llm_judge.engine import evaluate_case, evaluate_cases, score_with_judges
from llm_judge.io import load_cases
from llm_judge.providers import LLMProvider, LLMResponse, build_provider
from llm_judge.scorers import generate_answer, llm_score
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


def test_generate_answer_uses_plain_text_mode() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="", expected="A", chunks=["ctx"])

    result = generate_answer(case, PlainProvider())

    assert result.answer == "Generated answer."
    assert result.error is None


def test_llm_score_returns_error_on_unparseable_json() -> None:
    case = EvalCase(case_id="x", question="Q?", answer="A", expected="A", chunks=["ctx"])

    result = llm_score(case, BadJsonProvider(), parse_retries=0)

    assert result.verdict == "ERROR"
    assert result.score == 0.0


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
