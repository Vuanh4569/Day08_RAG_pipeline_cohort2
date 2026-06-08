from .local_rag_utils import chunk_text, load_markdown_documents


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive_character"
EMBEDDING_MODEL = "local-token-cosine"
EMBEDDING_DIM = 1024
VECTOR_STORE = "local-memory"


def load_documents() -> list[dict]:
    return load_markdown_documents()


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        for index, content in enumerate(chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(
                {
                    "content": content,
                    "metadata": {**doc.get("metadata", {}), "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        tokens = set(chunk["content"].lower().split())
        chunk["embedding"] = [1.0 if str(i) in tokens else 0.0 for i in range(EMBEDDING_DIM)]
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    return {"vector_store": VECTOR_STORE, "count": len(chunks)}


def run_pipeline():
    docs = load_documents()
    chunks = chunk_documents(docs)
    indexed = index_to_vectorstore(embed_chunks(chunks))
    print(f"Loaded {len(docs)} docs, created {len(chunks)} chunks, indexed {indexed['count']}.")


if __name__ == "__main__":
    run_pipeline()
