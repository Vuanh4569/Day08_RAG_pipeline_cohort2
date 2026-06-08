"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import math
import re
from collections import Counter

from .task4_chunking_indexing import (
    CHUNKS_INDEX_PATH,
    chunk_documents,
    embed_chunks,
    index_to_vectorstore,
    load_documents,
)

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def _ensure_corpus() -> list[dict]:
    """Load chunks from the local Task 4 index, creating it if needed."""
    global CORPUS
    if CORPUS:
        return CORPUS

    if not CHUNKS_INDEX_PATH.exists():
        docs = load_documents()
        chunks = chunk_documents(docs)
        embedded = embed_chunks(chunks)
        index_to_vectorstore(embedded)

    payload = json.loads(CHUNKS_INDEX_PATH.read_text(encoding="utf-8"))
    CORPUS = [
        {"content": chunk.get("content", ""), "metadata": chunk.get("metadata", {})}
        for chunk in payload.get("chunks", [])
    ]
    return CORPUS


def _tokenize(text: str) -> list[str]:
    """Simple Unicode word tokenizer for Vietnamese text without extra deps."""
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class SimpleBM25:
    """Small BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]

        doc_freq = Counter()
        for doc in tokenized_corpus:
            doc_freq.update(set(doc))

        total_docs = len(tokenized_corpus)
        self.idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for idx, freqs in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_len[idx] or 1
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]

    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized_corpus)
    except Exception:
        return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global BM25_INDEX

    if top_k <= 0:
        return []

    corpus = _ensure_corpus()
    if not corpus:
        return []

    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(corpus)

    tokenized_query = _tokenize(query)
    scores = list(BM25_INDEX.get_scores(tokenized_query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        results.append({
            "content": corpus[idx]["content"],
            "score": score,
            "metadata": corpus[idx]["metadata"],
        })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        preview = r["content"][:100].encode("ascii", errors="ignore").decode("ascii")
        print(f"[{r['score']:.3f}] {preview}...")
