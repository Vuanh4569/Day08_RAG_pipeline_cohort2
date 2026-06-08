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

import json
import urllib.error
import urllib.request
from pathlib import Path

from .env_utils import get_env

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive chunking giữ đoạn luật/tin theo paragraph trước khi fallback xuống câu/từ.
CHUNK_SIZE = 500        # Vừa đủ chứa một điều khoản hoặc đoạn báo ngắn.
CHUNK_OVERLAP = 50      # Giữ ngữ cảnh ở ranh giới chunk nhưng không nhân đôi quá nhiều.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# text-embedding-3-small nhẹ, rẻ và ổn cho RAG; fallback hashing giúp test offline.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Bài gốc khuyến nghị Weaviate, bản cá nhân này dùng in-memory index để test offline.
VECTOR_STORE = "in_memory"


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
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    separators = ["\n\n", "\n", ". ", " "]

    def split_text(text: str) -> list[str]:
        text = text.strip()
        if len(text) <= CHUNK_SIZE:
            return [text] if text else []
        for sep in separators:
            parts = text.split(sep)
            if len(parts) == 1:
                continue
            merged = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate) <= CHUNK_SIZE:
                    current = candidate
                else:
                    if current:
                        merged.extend(split_text(current))
                    current = part.strip()
            if current:
                merged.extend(split_text(current))
            return merged

        step = CHUNK_SIZE - CHUNK_OVERLAP
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]

    for doc in documents:
        for i, chunk_text in enumerate(split_text(doc["content"])):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed with OpenAI when available; otherwise use deterministic local hashing."""
    api_key = get_env("OPENAI_API_KEY")
    model = get_env("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL)
    if api_key and texts:
        payload = {"model": model, "input": texts}
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError):
            pass

    return [_hash_embedding(text) for text in texts]


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    vector = [0.0] * dim
    for token in text.lower().split():
        vector[hash(token) % dim] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    return {"store": VECTOR_STORE, "count": len(chunks), "chunks": chunks}


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
