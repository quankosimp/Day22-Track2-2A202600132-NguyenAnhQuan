"""Step 3: RAGAS evaluation for both prompt versions."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import ROOT_DIR, load_settings, openai_embedding_kwargs

settings = load_settings(require_secrets=True)

from langchain_openai import OpenAIEmbeddings

from qa_pairs import QA_PAIRS
from rag_utils import PROMPTS, build_vectorstore, create_llm, run_rag_once


REPORT_PATH = ROOT_DIR / "data" / "ragas_report.json"


def collect_rag_outputs(vectorstore, prompt_version: str) -> list[dict]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = create_llm(settings)
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag_once(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/50] {qa['question']}")
    return results


def build_ragas_dataset(rag_results: list[dict]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["reference"],
        )
        for row in rag_results
    ]
    return EvaluationDataset(samples=samples)


def _mean_metric(result, metric_name: str) -> float:
    values = result[metric_name]
    return float(np.nanmean(values))


def run_ragas_eval(rag_results: list[dict], version: str) -> dict:
    dataset = build_ragas_dataset(rag_results)
    eval_llm = create_llm(settings)
    eval_embeddings = OpenAIEmbeddings(**openai_embedding_kwargs(settings))
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    scores = {
        "faithfulness": _mean_metric(result, "faithfulness"),
        "answer_relevancy": _mean_metric(result, "answer_relevancy"),
        "context_recall": _mean_metric(result, "context_recall"),
        "context_precision": _mean_metric(result, "context_precision"),
    }
    print(f"\nRAGAS scores for {version}: {scores}")
    return scores


def print_comparison(scores: dict) -> None:
    print("\n" + "=" * 72)
    print("  RAGAS Comparison: V1 vs V2")
    print("=" * 72)
    print(f"{'Metric':<22} {'V1':>10} {'V2':>10} {'Winner':>12}")
    print("-" * 72)
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        v1 = scores["v1"][metric]
        v2 = scores["v2"][metric]
        winner = "v1" if v1 > v2 else "v2" if v2 > v1 else "tie"
        print(f"{metric:<22} {v1:>10.3f} {v2:>10.3f} {winner:>12}")

    if max(scores["v1"]["faithfulness"], scores["v2"]["faithfulness"]) >= 0.8:
        print("\nTarget met: faithfulness >= 0.8 for at least one prompt version")
    else:
        print("\nTarget not met: tune retrieval or prompts and rerun evaluation")


def main() -> None:
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore(settings)
    rag_outputs = {
        "v1": collect_rag_outputs(vectorstore, "v1"),
        "v2": collect_rag_outputs(vectorstore, "v2"),
    }
    scores = {
        "v1": run_ragas_eval(rag_outputs["v1"], "v1"),
        "v2": run_ragas_eval(rag_outputs["v2"], "v2"),
    }

    report = {"scores": scores, "samples": rag_outputs}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_comparison(scores)
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
