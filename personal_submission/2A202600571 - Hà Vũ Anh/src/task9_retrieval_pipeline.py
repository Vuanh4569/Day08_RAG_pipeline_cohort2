"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.
"""

import math
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def sigmoid(x: float) -> float:
    """Hàm sigmoid để chuyển đổi logits về đoạn [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.
    """
    # Step 1: Chạy semantic + lexical song song
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge bằng Reciprocal Rank Fusion (RRF)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        # Sử dụng sigmoid để đưa cross-encoder raw logits về khoảng [0, 1]
        for r in final_results:
            r["score"] = sigmoid(r["score"])
    else:
        # Nếu không rerank, scale RRF score lên [0, 1] dựa trên rank đầu tiên
        final_results = merged[:top_k]
        if final_results:
            max_s = final_results[0]["score"]
            if max_s > 0:
                for r in final_results:
                    r["score"] = r["score"] / max_s

    # Step 4: Check threshold → fallback
    # Nếu không có kết quả hoặc kết quả cao nhất nhỏ hơn ngưỡng điểm → Fallback sang PageIndex
    if not final_results or final_results[0]["score"] < score_threshold:
        best_score = final_results[0]["score"] if final_results else 0.0
        print(f"  ⚠ Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). Fallback → PageIndex")
        fallback_results = pageindex_search(query, top_k=top_k)
        return fallback_results

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "ca sĩ Chi Dân bị bắt vì sử dụng ma tuý",
        "xyzabc123nonsense"  # Kỳ vọng kích hoạt fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
