from collections import Counter

from .local_rag_utils import cosine_from_counters, tokenize


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    query_vector = Counter(tokenize(query))
    reranked = []
    for item in candidates:
        lexical_score = cosine_from_counters(query_vector, Counter(tokenize(item.get("content", ""))))
        original_score = float(item.get("score", 0.0))
        reranked.append({**item, "score": 0.7 * lexical_score + 0.3 * original_score})
    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    rrf_scores = {}
    content_map = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")
