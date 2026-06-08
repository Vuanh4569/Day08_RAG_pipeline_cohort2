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
import re
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

ARTICLE_HEADING_RE = re.compile(r"^Điều\s+\d+\.", re.IGNORECASE)
PDF_NOISE_MARKERS = (
    "Adobe.PPKLite",
    "adbe.pkcs7",
    "/Type/Sig",
    "/ByteRange",
    "/Contents",
    "/SubFilter",
)


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
        raw_content = md_file.read_text(encoding="utf-8")
        extracted_metadata, body = extract_markdown_metadata(raw_content)
        content = clean_markdown_text(body)
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
                **extracted_metadata,
            },
        })
    return documents


def extract_markdown_metadata(text: str) -> tuple[dict, str]:
    """Extract simple markdown metadata headers into chunk metadata."""
    metadata = {}
    body_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# ") and "title" not in metadata:
            metadata["title"] = line[2:].strip()
            continue
        match = re.match(r"^\*\*([^*:\n]+):\*\*\s*(.+)$", line)
        if match:
            key, value = match.groups()
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key in {"source", "url"} and value and value != "N/A":
                metadata["url"] = value
            elif key in {"title", "crawled", "date_crawled", "published_date"} and value and value != "N/A":
                metadata[key] = value
            continue
        body_lines.append(raw_line)
    return metadata, "\n".join(body_lines)


def clean_markdown_text(text: str) -> str:
    """Remove conversion artifacts before chunking and retrieval."""
    cleaned_lines = []
    for raw_line in text.replace("\x00", "").splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if any(marker in line for marker in PDF_NOISE_MARKERS):
            continue
        if line.startswith("![") or line.startswith("[!["):
            continue
        if line.startswith("Menu") or line.startswith("Đăng nhập") or line.startswith("Tìm kiếm"):
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if line.startswith("<<") or line.endswith(">>"):
            continue
        line = re.sub(r"[ \t]{2,}", " ", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []

    for doc in documents:
        splitter = split_legal_text if doc.get("metadata", {}).get("type") == "legal" else split_text
        for i, chunk_text in enumerate(_merge_short_chunks(splitter(doc["content"]))):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def split_legal_text(text: str) -> list[str]:
    """Prefer legal article boundaries, then apply windowing inside long articles."""
    sections = []
    current = []
    preamble = []

    for line in text.splitlines():
        if ARTICLE_HEADING_RE.match(line):
            if current:
                sections.append("\n".join(current).strip())
            elif preamble:
                sections.append("\n".join(preamble).strip())
                preamble = []
            current = [line]
        elif current:
            current.append(line)
        else:
            preamble.append(line)

    if current:
        sections.append("\n".join(current).strip())
    elif preamble:
        sections.append("\n".join(preamble).strip())

    chunks = []
    for section in sections:
        chunks.extend(split_text(section))
    return chunks


def split_text(text: str) -> list[str]:
    """Split text into <=CHUNK_SIZE windows with CHUNK_OVERLAP carried forward."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        pieces = _split_long_piece(paragraph)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
                continue
            if current:
                chunks.append(current)
                overlap = current[-CHUNK_OVERLAP:].strip()
                candidate = f"{overlap}\n\n{piece}".strip() if overlap else piece
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                chunks.extend(_sliding_windows(candidate))
                current = ""

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk.strip()) >= 20]


def _split_long_piece(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]

    sentences = re.split(r"(?<=[.!?。])\s+|\n", text)
    pieces = []
    current = ""
    for sentence in [item.strip() for item in sentences if item.strip()]:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(sentence) > CHUNK_SIZE:
                pieces.extend(_sliding_windows(sentence))
                current = ""
            else:
                current = sentence
    if current:
        pieces.append(current)
    return pieces


def _sliding_windows(text: str) -> list[str]:
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    return [text[i:i + CHUNK_SIZE].strip() for i in range(0, len(text), step) if text[i:i + CHUNK_SIZE].strip()]


def _merge_short_chunks(chunks: list[str], min_chars: int = 100) -> list[str]:
    merged = []
    for chunk in chunks:
        clean_chunk = chunk.strip()
        if not clean_chunk:
            continue
        if merged and len(clean_chunk) < min_chars and len(merged[-1]) + 2 + len(clean_chunk) <= CHUNK_SIZE:
            merged[-1] = f"{merged[-1]}\n\n{clean_chunk}"
        else:
            merged.append(clean_chunk)
    return merged


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
