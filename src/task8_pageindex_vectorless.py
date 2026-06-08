from .local_rag_utils import load_chunks, term_overlap_score


def upload_documents():
    return {"uploaded": len(load_chunks())}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    results = []
    for chunk in load_chunks():
        score = term_overlap_score(query, chunk["content"])
        if score > 0:
            results.append({**chunk, "score": float(score), "source": "pageindex"})

    if not results:
        results = [{**chunk, "score": 0.01, "source": "pageindex"} for chunk in load_chunks()[:top_k]]

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
