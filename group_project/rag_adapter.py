"""Adapter between the personal RAG pipeline and the group Gradio chatbot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


GROUP_DIR = Path(__file__).resolve().parent
DAY08_DIR = GROUP_DIR.parent
PERSONAL_ROOT = DAY08_DIR / "personal_submission"


def _find_personal_dir() -> Path:
    preferred = PERSONAL_ROOT / "2A202600802 - Phạm Đình Phúc"
    candidates = [preferred] if preferred.exists() else []
    if PERSONAL_ROOT.exists():
        candidates.extend(path for path in PERSONAL_ROOT.iterdir() if path.is_dir() and path != preferred)

    for candidate in candidates:
        src_dir = candidate / "src"
        if (src_dir / "task9_retrieval_pipeline.py").exists() and (src_dir / "task10_generation.py").exists():
            return candidate

    raise ModuleNotFoundError(
        "Cannot find a personal submission folder containing src/task9_retrieval_pipeline.py "
        "and src/task10_generation.py."
    )


PERSONAL_DIR = _find_personal_dir()

if str(PERSONAL_DIR) not in sys.path:
    sys.path.insert(0, str(PERSONAL_DIR))


def answer_question(
    question: str,
    history: list[tuple[str, str]] | None = None,
    top_k: int = 5,
    use_memory: bool = True,
    use_reranking: bool = True,
) -> dict[str, Any]:
    """Run the group chatbot pipeline and return answer plus source data."""
    clean_question = question.strip()
    if not clean_question:
        return {
            "answer": "Bạn hãy nhập câu hỏi trước nhé.",
            "sources": [],
            "source_markdown": "_Chưa có nguồn nào được dùng._",
            "retrieval_source": "none",
        }

    retrieve, generation = _load_personal_pipeline()
    enriched_query = _with_memory(clean_question, history or []) if use_memory else clean_question
    chunks = retrieve(enriched_query, top_k=top_k, use_reranking=use_reranking)
    answer = _generate_answer(clean_question, chunks, generation)

    return {
        "answer": answer,
        "sources": chunks,
        "source_markdown": format_sources_markdown(chunks),
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


def format_sources_markdown(chunks: list[dict]) -> str:
    """Format retrieved chunks for the Gradio source panel."""
    if not chunks:
        return "_Không tìm thấy source phù hợp._"

    lines = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        title = metadata.get("title", source)
        url = metadata.get("url", "")
        doc_type = metadata.get("type", "unknown")
        retrieval_source = chunk.get("source", "hybrid")
        score = float(chunk.get("score", 0.0))
        preview = " ".join(chunk.get("content", "").split())[:600]
        lines.append(
            f"### {index}. `{title}`\n"
            f"- Source file: `{source}`\n"
            f"- URL: {url or '`N/A`'}\n"
            f"- Type: `{doc_type}`\n"
            f"- Retrieval: `{retrieval_source}`\n"
            f"- Score: `{score:.4f}`\n\n"
            f"{preview}\n"
        )
    return "\n---\n".join(lines)


def _load_personal_pipeline():
    from src import task10_generation
    from src.task9_retrieval_pipeline import retrieve

    return retrieve, task10_generation


def _generate_answer(question: str, chunks: list[dict], generation_module) -> str:
    """Generate a citation-focused answer using OpenAI, with local fallback."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    ordered_chunks = generation_module.reorder_for_llm(chunks)
    context = generation_module.format_context(ordered_chunks)

    messages = [
        {"role": "system", "content": generation_module.SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    answer = _chat_completion_with_openai(messages, generation_module)
    if answer:
        return _ensure_citations(answer, ordered_chunks)

    answer = _chat_completion_with_groq(messages, generation_module)
    if answer:
        return _ensure_citations(answer, ordered_chunks)

    first = ordered_chunks[0]
    source = first.get("metadata", {}).get("source", "Nguồn nội bộ")
    snippet = " ".join(first.get("content", "").split())[:700]
    return f"{snippet} [{source}]"


def _ensure_citations(answer: str, chunks: list[dict]) -> str:
    """Append source references when the model forgets bracket citations."""
    if "[" in answer and "]" in answer:
        return answer

    sources = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        title = metadata.get("title") or metadata.get("source", "unknown")
        url = metadata.get("url", "")
        source = f"{title}: {url}" if url else title
        if source not in sources:
            sources.append(source)
        if len(sources) >= 3:
            break

    if not sources:
        return answer

    citation_line = "; ".join(f"[{source}]" for source in sources)
    return f"{answer}\n\nNguồn đã dùng: {citation_line}"


def _chat_completion_with_openai(messages: list[dict], generation_module) -> str:
    try:
        from openai import OpenAI
        from src.env_utils import get_env

        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            return ""

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=get_env("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=generation_module.TEMPERATURE,
            top_p=generation_module.TOP_P,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _chat_completion_with_groq(messages: list[dict], generation_module) -> str:
    try:
        from openai import OpenAI
        from src.env_utils import get_env

        api_key = get_env("GROQ_API_KEY")
        if not api_key:
            return ""

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=get_env("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=messages,
            temperature=generation_module.TEMPERATURE,
            top_p=generation_module.TOP_P,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _with_memory(question: str, history: list[tuple[str, str]]) -> str:
    """Append the last turns so follow-up questions have enough context."""
    normalized = []
    for user_message, assistant_message in history[-3:]:
        if user_message:
            normalized.append(f"user: {user_message[:300]}")
        if assistant_message:
            normalized.append(f"assistant: {assistant_message[:300]}")

    if not normalized:
        return question

    context = "\n".join(normalized[-6:])
    return f"Conversation context:\n{context}\n\nCurrent question: {question}"


if __name__ == "__main__":
    demo = answer_question("Luật phòng chống ma túy quy định những hình thức cai nghiện nào?")
    print(demo["answer"])
    print("\nSources:\n", demo["source_markdown"][:1000])
