"""Shared configuration for the Day 22 LangSmith RAG lab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"


@dataclass(frozen=True)
class Settings:
    langchain_tracing_v2: str
    langchain_api_key: str
    langchain_project: str
    langchain_endpoint: str
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    openai_embedding_model: str


def load_settings(require_secrets: bool = False) -> Settings:
    load_dotenv(ENV_PATH)

    langchain_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY", "")
    if langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGSMITH_API_KEY"] = langchain_api_key

    settings = Settings(
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true"),
        langchain_api_key=langchain_api_key,
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-rag-lab"),
        langchain_endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    missing = []
    if not settings.langchain_api_key:
        missing.append("LANGCHAIN_API_KEY")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if require_secrets and missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required .env values: {joined}")

    return settings


def openai_chat_kwargs(settings: Settings) -> dict:
    kwargs = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": 0,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return kwargs


def openai_embedding_kwargs(settings: Settings) -> dict:
    kwargs = {
        "model": settings.openai_embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return kwargs


def main() -> None:
    settings = load_settings(require_secrets=False)
    missing = []
    if not settings.langchain_api_key:
        missing.append("LANGCHAIN_API_KEY")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        print("Config loaded, but these .env values are still empty:")
        for name in missing:
            print(f"   - {name}")
    else:
        print("Config loaded successfully")
    print(f"   LangSmith project : {settings.langchain_project}")
    print(f"   OpenAI endpoint   : {settings.openai_base_url or 'default OpenAI endpoint'}")
    print(f"   Default LLM model : {settings.openai_model}")
    print(f"   Embedding model   : {settings.openai_embedding_model}")


if __name__ == "__main__":
    main()
