from collections import Counter

from .local_rag_utils import cosine_from_counters, load_chunks, tokenize


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    query_vector = Counter(tokenize(query))
    results = []
    for chunk in load_chunks():
        score = cosine_from_counters(query_vector, Counter(tokenize(chunk["content"])))
        if score > 0:
            results.append({**chunk, "score": float(score)})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
