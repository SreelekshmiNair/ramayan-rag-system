from collections.abc import Sequence

from langchain_core.documents import Document
from llama_cpp import Llama

from .llm import BASELINE_SYSTEM_MESSAGE, RAG_SYSTEM_MESSAGE, format_instruct_prompt


def _call_llm(
    llm: Llama,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=1.1,
        stop=["</s>"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def baseline_answer(
    llm: Llama,
    question: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    prompt = format_instruct_prompt(BASELINE_SYSTEM_MESSAGE, question)
    return _call_llm(
        llm,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def format_context(documents: Sequence[Document]) -> str:
    sections: list[str] = []
    for index, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        source = doc.metadata.get("source", "unknown source")
        sections.append(
            f"[Chunk {index} | {page_label} | {source}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(sections)


def rag_answer(
    llm: Llama,
    vectorstore,
    question: str,
    *,
    retrieval_k: int = 3,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 50,
) -> tuple[str, list[Document]]:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )
    documents = retriever.invoke(question)
    context = format_context(documents)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer using only the context above."
    )
    prompt = format_instruct_prompt(RAG_SYSTEM_MESSAGE, user_message)
    answer = _call_llm(
        llm,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return answer, documents


def summarize_sources(documents: Sequence[Document]) -> str:
    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        preview = " ".join(doc.page_content.split())[:220]
        lines.append(f"{index}. {page_label}: {preview}")
    return "\n".join(lines)
