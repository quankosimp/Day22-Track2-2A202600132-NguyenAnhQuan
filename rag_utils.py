"""Reusable RAG helpers for the Day 22 lab scripts."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    ROOT_DIR,
    Settings,
    openai_chat_kwargs,
    openai_embedding_kwargs,
)


KNOWLEDGE_BASE_PATH = ROOT_DIR / "data" / "knowledge_base.txt"

SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer the user's question using ONLY the "
    "provided context. Keep your answer concise (2-4 sentences). If the context "
    "does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V1), ("human", "{question}")]
)
PROMPT_V2 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V2), ("human", "{question}")]
)
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

RAG_PROMPT = PROMPT_V1


def create_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(**openai_chat_kwargs(settings))


def create_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(**openai_embedding_kwargs(settings))


def load_knowledge_base(path: Path = KNOWLEDGE_BASE_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")
    return path.read_text(encoding="utf-8")


def split_knowledge_base(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def build_vectorstore(settings: Settings) -> FAISS:
    text = load_knowledge_base()
    chunks = split_knowledge_base(text)
    embeddings = create_embeddings(settings)
    metadatas = [{"source": "knowledge_base.txt", "chunk": i} for i in range(len(chunks))]
    print(f"Split knowledge base into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore: FAISS, settings: Settings, prompt=RAG_PROMPT):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = create_llm(settings)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def run_rag_once(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": "\n\n".join(contexts), "question": question}
    )
    return {"answer": answer, "contexts": contexts}
