"""
Task 5 — Semantic Search Module.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths relative to this file
CURRENT_DIR = Path(__file__).parent
LOCAL_DB_PATH = CURRENT_DIR.parent / "data" / "vector_store.json"
ROOT_DB_PATH = CURRENT_DIR.parent.parent.parent / "data" / "vector_store.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vector_db() -> list[dict]:
    """Tải vector database từ file JSON."""
    db_path = LOCAL_DB_PATH
    if not db_path.exists():
        db_path = ROOT_DB_PATH
        
    if not db_path.exists():
        print(f"⚠ Không tìm thấy vector store tại: {LOCAL_DB_PATH} hoặc {ROOT_DB_PATH}")
        return []
        
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Tính toán cosine similarity giữa hai vector."""
    vec1 = np.array(a)
    vec2 = np.array(b)
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


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
    chunks = load_vector_db()
    if not chunks:
        return []

    # Load embedding model and encode query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode(query).tolist()

    results = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        
        score = cosine_similarity(query_vector, embedding)
        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk.get("metadata", {})
        })

    # Sắp xếp giảm dần theo similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ Chi Dân bị bắt"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        res = semantic_search(q, top_k=3)
        for r in res:
            print(f"[{r['score']:.4f}] {r['content'][:150]}...")
