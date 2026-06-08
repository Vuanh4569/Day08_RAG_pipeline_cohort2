"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from __future__ import annotations

import math
from collections import Counter

from .offline_retrieval import cosine_from_counters, get_chunks, tokenize
from .task4_chunking_indexing import embed_chunks, embed_texts


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    chunks = get_chunks()
    if not chunks:
        return []

    query_embedding = embed_texts([query])[0]
    embedded_chunks = embed_chunks(chunks)
    query_counter = Counter(tokenize(query))

    results = []
    for chunk in embedded_chunks:
        score = _cosine(query_embedding, chunk.get("embedding", []))
        if score <= 0:
            score = cosine_from_counters(query_counter, Counter(tokenize(chunk["content"])))
        if score > 0:
            results.append({
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk["metadata"],
            })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
