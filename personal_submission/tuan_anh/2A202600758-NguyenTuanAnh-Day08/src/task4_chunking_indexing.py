"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import hashlib
import json
import math
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CHUNKS_INDEX_PATH = INDEX_DIR / "chunks.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn recursive character chunking vì bộ dữ liệu gồm 2 loại nguồn khác nhau:
# văn bản luật được convert từ PDF và bài báo được convert từ JSON. Sau khi
# convert, heading trong file luật có thể không đều, nên MarkdownHeaderSplitter
# dễ tách sai. Recursive chunking ưu tiên tách theo đoạn, dòng, câu rồi mới
# tách theo ký tự, vì vậy ổn định hơn cho dữ liệu lẫn legal/news.
CHUNK_SIZE = 500        # Đủ ngắn để truy hồi đúng đoạn, không kéo quá nhiều nhiễu.
CHUNK_OVERLAP = 50      # Giữ lại ngữ cảnh giữa 2 chunk, hữu ích cho điều luật dài.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn embedding local-hashing-v1 để bài lab chạy ổn định trên máy cá nhân,
# không cần tải model nặng từ HuggingFace, không cần GPU và không phụ thuộc API.
# Vector 384 chiều mô phỏng kích thước của all-MiniLM-L6-v2, đủ để Task 5
# semantic search tính cosine similarity trên index local.
EMBEDDING_MODEL = "local-hashing-v1"
EMBEDDING_DIM = 384

# Chọn local-json thay cho Weaviate trong bản nộp cá nhân để không cần Docker
# hoặc Weaviate Cloud. Index được lưu ở data/index/chunks.json, dễ kiểm tra,
# dễ chạy lại và các Task 5-9 có thể đọc trực tiếp.
VECTOR_STORE = "local-json"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        try:
            doc_type = md_file.relative_to(STANDARDIZED_DIR).parts[0]
        except IndexError:
            doc_type = "unknown"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })

    return documents


def _split_long_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into pieces that never exceed chunk_size characters."""
    parts = []
    remaining = text.strip()

    while len(remaining) > chunk_size:
        split_at = -1
        for separator in ("\n\n", "\n", ". ", " "):
            idx = remaining.rfind(separator, 0, chunk_size + 1)
            if idx > 0:
                split_at = idx + len(separator)
                break

        if split_at <= 0:
            split_at = chunk_size

        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)

    return [part for part in parts if part]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []

    for doc in documents:
        raw_splits = _split_long_text(doc["content"], CHUNK_SIZE)
        for i, split in enumerate(raw_splits):
            if i > 0 and CHUNK_OVERLAP > 0:
                overlap = raw_splits[i - 1][-CHUNK_OVERLAP:]
                chunk_text = f"{overlap} {split}".strip()
                if len(chunk_text) > CHUNK_SIZE:
                    chunk_text = chunk_text[-CHUNK_SIZE:]
            else:
                chunk_text = split

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc.get("metadata", {}),
                    "chunk_index": i,
                    "chunking_method": CHUNKING_METHOD,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                },
            })

    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Create a deterministic normalized bag-of-words hashing embedding."""
    vector = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embedded = []
    for chunk in chunks:
        item = dict(chunk)
        item["embedding"] = _hash_embedding(item["content"])
        item["metadata"] = {
            **item.get("metadata", {}),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
        }
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store": VECTOR_STORE,
        },
        "chunks": chunks,
    }
    CHUNKS_INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return CHUNKS_INDEX_PATH


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
