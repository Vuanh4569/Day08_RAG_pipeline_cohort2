"""
Task 7 — Reranking Module.
"""

import os
import requests
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# We can fall back to using sentence-transformers' CrossEncoder or embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"  # Very lightweight (~30MB)


def cosine_sim(a, b):
    vec1 = np.array(a)
    vec2 = np.array(b)
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank candidates sử dụng cross-encoder model."""
    if not candidates:
        return []

    # Try Jina AI API if key is present
    jina_key = os.getenv("JINA_API_KEY", "")
    if jina_key and not jina_key.startswith("jina_xxx"):
        print("Using Jina AI Reranker API...")
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                return [
                    {**candidates[r["index"]], "score": r["relevance_score"]}
                    for r in reranked
                ]
        except Exception as e:
            print(f"Jina API error: {e}. Falling back to local model.")

    # Try local cross-encoder model
    try:
        from sentence_transformers import CrossEncoder
        print(f"Using local CrossEncoder: {CROSS_ENCODER_MODEL}...")
        model = CrossEncoder(CROSS_ENCODER_MODEL)
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        
        # Norm scores to [0, 1] range if needed or keep raw logits
        results = []
        for c, score in zip(candidates, scores):
            # Convert to standard float
            results.append({**c, "score": float(score)})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Local CrossEncoder error: {e}. Falling back to semantic similarity reranker.")
        
    # Final fallback: Semantic Similarity (bi-encoder similarity) using Bge/MiniLM model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_emb = model.encode(query).tolist()
        
        results = []
        for c in candidates:
            doc_emb = c.get("embedding")
            if not doc_emb:
                doc_emb = model.encode(c["content"]).tolist()
            score = cosine_sim(query_emb, doc_emb)
            results.append({**c, "score": score})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Bi-encoder fallback error: {e}. Falling back to basic score sorting.")
        results = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse."""
    if not candidates:
        return []

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Ensure all candidates have embeddings
    for c in candidates:
        if "embedding" not in c or not c["embedding"]:
            c["embedding"] = model.encode(c["content"]).tolist()

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score formula
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    # Return the selected items with their MMR score
    results = []
    for idx in selected:
        item = candidates[idx].copy()
        # compute clean relevance score for output
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker."""
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sắp xếp theo điểm số RRF
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # MMR need query_embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_emb = model.encode(query).tolist()
        return rerank_mmr(query_emb, candidates, top_k)
    elif method == "rrf":
        # For simplicity, if RRF is called with single flat list, wrap it.
        # RRF expects list of ranked lists.
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    print("Testing cross encoder...")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
