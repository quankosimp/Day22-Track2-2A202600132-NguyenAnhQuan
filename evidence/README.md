# Evidence Checklist

## Code Files

- [x] `01_langsmith_rag_pipeline.py` runs and generated LangSmith traces.
- [x] `02_prompt_hub_ab_routing.py` runs, pushes/pulls 2 prompt versions, and logs A/B routing.
- [x] `03_ragas_evaluation.py` runs and saves `data/ragas_report.json`.
- [x] `04_guardrails_validator.py` runs PII and JSON validation demos.
- [x] `run_all.py` exists for sequential or per-step execution.
- [x] `config.py`, `qa_pairs.py`, `rag_utils.py`, and `data/knowledge_base.txt` exist.
- [x] Python files pass `py_compile`.

## Evidence Files

- [x] `01_langsmith_traces.png`: LangSmith project run tab showing traces.
- [x] `02_prompt_hub.png`: Prompt Hub showing both prompt versions.
- [x] `02_ab_routing_log.txt`: Console output from A/B routing.
- [x] `03_ragas_scores.png`: Terminal screenshot showing the V1 vs V2 comparison table.
- [x] `03_ragas_report.json`: Copy of `data/ragas_report.json`.
- [x] `04_pii_demo_log.txt`: PII validator demo output.
- [x] `04_json_demo_log.txt`: JSON formatter demo output.

## RAGAS Result

- [x] `v1` faithfulness is above 0.8.
- [x] Both prompt versions were evaluated.
- [x] All required metrics are present: faithfulness, answer relevancy, context recall, context precision.

## Submission

- [x] `.env` is ignored by git.
- [x] No API key pattern found in tracked source/evidence scan.
- [ ] Public GitHub repo URL submitted on the course portal.
- [ ] LangSmith project URL submitted on the course portal.
