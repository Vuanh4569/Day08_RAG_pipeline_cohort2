import math
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def infer_doc_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_markdown_documents() -> list[dict]:
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(PROJECT_DIR)),
                    "type": infer_doc_type(md_file),
                },
            }
        )
    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(clean):
            break
        start += step
    return chunks


def load_chunks(chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    chunks = []
    for doc in load_markdown_documents():
        for index, content in enumerate(chunk_text(doc["content"], chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "content": content,
                    "metadata": {**doc["metadata"], "chunk_index": index},
                }
            )
    return chunks


def cosine_from_counters(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def term_overlap_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)
