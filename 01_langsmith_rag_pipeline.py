"""Step 1: LangSmith-instrumented RAG pipeline."""

from __future__ import annotations

from config import load_settings

settings = load_settings(require_secrets=True)

from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS
from rag_utils import build_rag_chain, build_vectorstore


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main() -> None:
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore(settings)
    chain, _ = build_rag_chain(vectorstore, settings)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {answer[:180]}\n")

    print(
        f"{len(SAMPLE_QUESTIONS)} traces sent to LangSmith project "
        f"'{settings.langchain_project}'."
    )
    print("Open https://smith.langchain.com to capture evidence.")


if __name__ == "__main__":
    main()
