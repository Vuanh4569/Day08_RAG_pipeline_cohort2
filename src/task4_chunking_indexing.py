from personal_submission.vu_anh.src.task4_chunking_indexing import (
    load_documents,
    chunk_documents,
    embed_chunks,
    index_to_vectorstore,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_METHOD,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    VECTOR_STORE,
    run_pipeline
)

if __name__ == "__main__":
    run_pipeline()
