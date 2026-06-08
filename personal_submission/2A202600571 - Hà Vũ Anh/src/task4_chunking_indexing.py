"""
Task 4 — Chunking & Indexing vào Vector Store.
"""

import json
from pathlib import Path

# Config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
VECTOR_STORE = "local_file"

# Paths
CURRENT_DIR = Path(__file__).parent
STANDARDIZED_DIR = CURRENT_DIR.parent / "data" / "standardized"
ROOT_STANDARDIZED_DIR = CURRENT_DIR.parent.parent.parent / "data" / "standardized"

VECTOR_DB_PATH = CURRENT_DIR.parent / "data" / "vector_store.json"
ROOT_VECTOR_DB_PATH = CURRENT_DIR.parent.parent.parent / "data" / "vector_store.json"


def load_documents_for_path(path: Path) -> list[dict]:
    """Đọc toàn bộ markdown files từ thư mục chỉ định."""
    documents = []
    if not path.exists():
        return documents
        
    for md_file in path.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ STANDARDIZED_DIR."""
    # We load local documents
    return load_documents_for_path(STANDARDIZED_DIR)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents sử dụng RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks sử dụng sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào file JSON đóng vai trò vector store."""
    # Đảm bảo thư mục cha tồn tại
    VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save locally
    with open(VECTOR_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved vector database to {VECTOR_DB_PATH}")

    # Save to root data directory as well
    ROOT_VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROOT_VECTOR_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved vector database to {ROOT_VECTOR_DB_PATH}")


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} local documents")

    # If local documents is empty, try loading root documents as fallback
    if not docs:
        docs = load_documents_for_path(ROOT_STANDARDIZED_DIR)
        print(f"✓ Loaded {len(docs)} root documents (fallback)")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to local vector store")


if __name__ == "__main__":
    run_pipeline()
