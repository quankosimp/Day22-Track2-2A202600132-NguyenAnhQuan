"""Step 2: Prompt Hub versioning and deterministic A/B routing."""

from __future__ import annotations

import hashlib
from collections import Counter

from config import load_settings

settings = load_settings(require_secrets=True)

from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS
from rag_utils import (
    PROMPT_V1,
    PROMPT_V2,
    build_vectorstore,
    create_llm,
    run_rag_once,
)


PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"


def push_prompts_to_hub(client: Client) -> None:
    prompts = [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 concise RAG prompt for Day 22 lab"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 structured RAG prompt for Day 22 lab"),
    ]
    for name, prompt, description in prompts:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"Pushed {name}: {url}")
        except Exception as exc:
            print(f"Prompt push skipped for {name}: {exc}")


def pull_prompts_from_hub(client: Client) -> dict:
    local = {PROMPT_V1_NAME: PROMPT_V1, PROMPT_V2_NAME: PROMPT_V2}
    pulled = {}
    for name, fallback in local.items():
        try:
            pulled[name] = client.pull_prompt(name)
            print(f"Pulled {name} from LangSmith Prompt Hub")
        except Exception as exc:
            pulled[name] = fallback
            print(f"Using local fallback for {name}: {exc}")
    return pulled


def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    out = run_rag_once(retriever, llm, prompt, question)
    return {"question": question, "answer": out["answer"], "version": version}


def main() -> None:
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=settings.langchain_api_key)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = create_llm(settings)

    counts = Counter()
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        counts[version_tag] += 1
        result = ask_ab(retriever, llm, prompts[version_key], question, version_tag)
        print(f"[{i:02d}] [prompt-{version_tag}] {result['question']}")
        print(f"     {result['answer'][:160]}\n")

    print(f"Routing summary: prompt-v1={counts['v1']}, prompt-v2={counts['v2']}")
    print(f"Additional traces sent to LangSmith project '{settings.langchain_project}'.")


if __name__ == "__main__":
    main()
