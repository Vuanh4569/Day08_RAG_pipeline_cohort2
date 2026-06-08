"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
import math

from .task4_chunking_indexing import (
    CHUNKS_INDEX_PATH,
    _hash_embedding,
    embed_chunks,
    chunk_documents,
    index_to_vectorstore,
    load_documents,
)


def _ensure_index() -> list[dict]:
    """Load local vector index, creating it from standardized markdown if needed."""
    if not CHUNKS_INDEX_PATH.exists():
        docs = load_documents()
        chunks = chunk_documents(docs)
        embedded = embed_chunks(chunks)
        index_to_vectorstore(embedded)

    payload = json.loads(CHUNKS_INDEX_PATH.read_text(encoding="utf-8"))
    return payload.get("chunks", [])


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity for normalized or non-normalized vectors."""
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    chunks = _ensure_index()
    query_embedding = _hash_embedding(query)

    results = []
    for chunk in chunks:
        score = _cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append({
            "content": chunk.get("content", ""),
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
