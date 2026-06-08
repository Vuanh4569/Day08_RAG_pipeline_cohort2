"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # Offline lexical reranker; replace with Jina for demo if needed.


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    retrieval_query = expand_query(query)
    dense_results = semantic_search(retrieval_query, top_k=top_k * 2)
    sparse_results = lexical_search(retrieval_query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    final_results = rerank(retrieval_query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking else merged[:top_k]
    for item in final_results:
        item["source"] = "hybrid"

    if not final_results or final_results[0].get("score", 0.0) < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


def expand_query(query: str) -> str:
    """Add domain synonyms that appear in Vietnamese legal documents."""
    lowered = query.lower()
    expansions = []
    if "cai nghiện" in lowered:
        expansions.extend([
            "biện pháp cai nghiện ma túy",
            "cai nghiện ma túy tự nguyện",
            "cai nghiện ma túy bắt buộc",
            "Điều 28",
            "Điều 29",
        ])
    if "hình thức" in lowered:
        expansions.append("biện pháp")
    if "nghệ sĩ" in lowered or "báo chí" in lowered:
        expansions.extend(["ca sĩ", "diễn viên", "người mẫu", "rapper", "ma túy"])
    return " ".join([query, *expansions])


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
