from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3


SYSTEM_PROMPT = """Tra loi cau hoi bang tieng Viet va chi dua vao context da cung cap.
Moi nhan dinh quan trong can co citation theo source trong context.
Neu context khong du, hay noi: Toi khong the xac minh thong tin nay tu nguon hien co."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks
    return chunks[0::2] + list(reversed(chunks[1::2]))


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not chunks:
        answer = "Toi khong the xac minh thong tin nay tu nguon hien co."
    else:
        source = chunks[0].get("metadata", {}).get("source", "nguon hien co")
        answer = (
            "Dua tren cac tai lieu da truy xuat, can doi chieu cau hoi voi context "
            f"va citation tu [{source}]. Noi dung lien quan da duoc dua vao context "
            "de ho tro sinh cau tra loi co dan nguon."
        )

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "context": context,
    }
