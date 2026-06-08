"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# OpenRouter-compatible model config. The "openrouter/free" router selects a
# currently available free model, which is useful for classroom/lab runs.
LLM_MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL", "openrouter/free")
LLM_BASE_URL = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or metadata.get("filename") or f"Source {i}"
        year = metadata.get("year", "n.d.")
        doc_type = metadata.get("type", "unknown")
        content = chunk.get("content", "")
        context_parts.append(
            f"[Document {i} | Source: {source} | Year: {year} | Type: {doc_type}]\n"
            f"{content}\n"
        )
    return "\n---\n".join(context_parts)


def _source_label(chunk: dict, index: int = 1) -> str:
    """Build a compact citation label like [source, year]."""
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or metadata.get("filename") or f"Source {index}"
    year = metadata.get("year") or metadata.get("published_at") or "n.d."
    return f"[{source}, {year}]"


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    """
    Deterministic fallback answer when an LLM/API key is unavailable.

    It keeps Task 10 runnable offline while still following the lab rule:
    answer only from retrieved context and include citation brackets.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    claims = []
    for i, chunk in enumerate(chunks[:3], 1):
        content = " ".join(chunk.get("content", "").split())
        if not content:
            continue
        snippet = content[:260].rstrip()
        claims.append(f"- {snippet} {_source_label(chunk, i)}")

    if not claims:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    return (
        f"Dựa trên các nguồn truy hồi được cho câu hỏi: {query}\n"
        + "\n".join(claims)
    )


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not context.strip():
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none",
        }

    user_message = f"""Context:
{context}

---

Question: {query}"""

    try:
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY/OPENAI_API_KEY")

        client_kwargs = {"api_key": api_key}
        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=512,
        )

        answer = response.choices[0].message.content
        if not answer:
            answer = _fallback_answer(query, chunks)
    except Exception as exc:
        answer = _fallback_answer(query, chunks)
        if answer == "Tôi không thể xác minh thông tin này từ nguồn hiện có":
            answer += f" (LLM error: {exc})"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        safe_q = q.encode("ascii", errors="ignore").decode("ascii")
        print(f"\n{'='*70}")
        print(f"Q: {safe_q}")
        print("=" * 70)
        result = generate_with_citation(q)
        safe_answer = result["answer"].encode("ascii", errors="ignore").decode("ascii")
        print(f"\nA: {safe_answer}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
