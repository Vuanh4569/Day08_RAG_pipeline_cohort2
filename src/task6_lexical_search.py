import math
from collections import Counter

from .local_rag_utils import load_chunks, tokenize


CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    tokenized = [tokenize(doc["content"]) for doc in corpus]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avg_len": avg_len, "n": len(corpus)}


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = CORPUS or load_chunks()
    index = build_bm25_index(corpus)
    query_terms = tokenize(query)
    k1 = 1.5
    b = 0.75
    results = []

    for doc, tokens in zip(corpus, index["tokenized"]):
        counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = index["doc_freq"][term]
            idf = math.log(1 + (index["n"] - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(1, index["avg_len"]))
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            results.append({**doc, "score": float(score)})

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
