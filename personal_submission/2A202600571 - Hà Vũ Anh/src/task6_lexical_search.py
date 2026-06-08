"""
Task 6 — Lexical Search Module (BM25).
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi

# Paths relative to this file
CURRENT_DIR = Path(__file__).parent
LOCAL_DB_PATH = CURRENT_DIR.parent / "data" / "vector_store.json"
ROOT_DB_PATH = CURRENT_DIR.parent.parent.parent / "data" / "vector_store.json"


def load_corpus() -> list[dict]:
    """Tải corpus từ vector store JSON."""
    db_path = LOCAL_DB_PATH
    if not db_path.exists():
        db_path = ROOT_DB_PATH
        
    if not db_path.exists():
        print(f"⚠ Không tìm thấy vector store tại: {LOCAL_DB_PATH} hoặc {ROOT_DB_PATH}")
        return []
        
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây dựng BM25 index từ corpus."""
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

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
    corpus = load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy các chỉ số có điểm số cao nhất
    results = []
    for idx, score in enumerate(scores):
        results.append({
            "content": corpus[idx]["content"],
            "score": float(score),
            "metadata": corpus[idx].get("metadata", {})
        })

    # Sắp xếp giảm dần theo điểm số BM25
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "hình phạt ma tuý",
        "Chi Dân"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        res = lexical_search(q, top_k=3)
        for r in res:
            print(f"[{r['score']:.4f}] {r['content'][:150]}...")
