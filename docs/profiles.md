# Input Profiles

Profiles map external benchmark schemas to the internal `EvalCase` shape.

Internal fields:

- `case_id`
- `question`
- `answer`
- `expected`
- `expected_facts`
- `chunks`
- `reference_contexts`
- `settings`
- `metadata`

## Supported Profiles

| Profile | Intended Input |
|---|---|
| `default` | Project-local JSONL with flexible aliases. |
| `ragas` | RAGAS-style `user_input`, `response`, `reference`, `retrieved_contexts`. |
| `locomo` | Long conversational memory benchmark outputs. |
| `longbench` | LongBench-like `input`, `prediction`, `answers`, `context`. |
| `langbench` | LongBench-like alias profile. |
| `pgraggraph-e2e` | pg-raggraph e2e result rows with nested `cell.*`. |
| `benchmark-results` | Generic result rows with `question`, generated `answer`, `gold`. |
| `graphrag-bench` | YAML question sets with `gold_answer`, `required_facts`, `expected_substring`. |
| `hotpotqa` | HotpotQA raw or prediction records. |
| `musique` | MuSiQue raw or prediction records. |
| `twowiki` | 2WikiMultiHopQA raw or prediction records. |
| `multihop-rag` | MultiHop-RAG records. |
| `chunkshop-e1e8` | Chunkshop E1-E8 records. |

## Chunkshop E1-E8 Expected Fields

Accepted aliases:

- ID: `id`, `case_id`, `qid`, `question_id`
- Question: `question`, `query`, `input`
- Answer: `answer`, `generated_answer`, `llm_answer`, `response`, `prediction`
- Expected answer: `gold_answer`, `expected`, `reference`, `answer_key`
- Required facts: `required_facts`, `expected_facts`, `facts`
- Chunks/context: `retrieved_full_context`, `retrieved_context`, `retrieved_chunks`, `chunks`, `summarized_answer_context`, `summary_context`, `context`
- Full/oracle context for generated gold answers: `reference_context`, `reference_contexts`, `oracle_context`, `full_context`, `full_data`
- Settings: `settings`, `config`, `config_label`, `experiment`, `strategy`, `retrievable`, `token_counts`, `retrieval_mode`, `chunker`

Example:

```json
{"id":"e1","question":"What changed?","gold_answer":"The parser was fixed.","required_facts":["Parser fix is identified."],"retrieved_chunks":["The parser was fixed in v2."],"config_label":"E1","retrievable":true,"token_counts":{"retrieved_context":42}}
```

## Raw Question Files

Some profiles can load raw question files that do not contain generated answers. These are useful for normalizing datasets, but answer accuracy requires either:

- a prediction field in the same record, or
- `--generate-answer` with retrieved chunks/context.

If a dataset has no gold/reference answer, provide full/oracle context and use `--generate-expected` before judging. The generated reference is stored as `expected`, while acceptable aliases are preserved in metadata for semantic judging.
