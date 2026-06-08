"""Small offline retrieval helpers used when external services are unavailable."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .task4_chunking_indexing import chunk_documents, load_documents


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d")
    return TOKEN_RE.findall(without_marks)


def cosine_from_counters(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def get_chunks() -> list[dict]:
    documents = load_documents()
    return chunk_documents(documents)


def keyword_score(query: str, content: str) -> float:
    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    content_counts = Counter(content_tokens)
    overlap = sum(content_counts[token] for token in set(query_tokens))
    coverage = len(set(query_tokens) & set(content_tokens)) / len(set(query_tokens))
    density = overlap / max(len(content_tokens), 1)
    return coverage + density
