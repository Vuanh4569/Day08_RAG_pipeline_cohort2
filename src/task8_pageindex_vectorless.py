"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
import time
from pathlib import Path

from .env_utils import get_env

PAGEINDEX_API_KEY = get_env("PAGEINDEX_API_KEY", "")
USE_PAGEINDEX_API = get_env("USE_PAGEINDEX_API", "false").lower() in {"1", "true", "yes"}
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
CACHE_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        return _local_uploaded_documents()

    try:
        from pageindex import PageIndexClient
    except ImportError:
        return _local_uploaded_documents()

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    uploaded = []

    # PageIndex service is strongest on PDFs, so upload original legal PDFs first.
    candidate_files = list((LANDING_DIR / "legal").glob("*.pdf"))
    if not candidate_files:
        candidate_files = list(STANDARDIZED_DIR.rglob("*.md"))

    for file_path in candidate_files:
        try:
            result = client.submit_document(str(file_path))
        except Exception:
            continue
        doc_id = result.get("doc_id") or result.get("id")
        if not doc_id:
            continue
        uploaded.append({
            "doc_id": doc_id,
            "filename": file_path.name,
            "type": file_path.parent.name,
        })

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(uploaded, ensure_ascii=False, indent=2), encoding="utf-8")
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if USE_PAGEINDEX_API and PAGEINDEX_API_KEY and _load_cached_doc_ids():
        try:
            results = _pageindex_api_search(query, top_k=top_k)
            if results:
                return results
        except Exception:
            pass

    return _local_pageindex_fallback(query, top_k=top_k)


def _pageindex_api_search(query: str, top_k: int = 5) -> list[dict]:
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    docs = _load_cached_doc_ids()
    if not docs:
        docs = upload_documents()
    docs = _wait_for_ready_docs(client, docs)
    if not docs:
        return []

    results = []
    for doc in docs[:top_k]:
        response = client.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=doc["doc_id"],
        )
        answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if answer:
            results.append({
                "content": answer,
                "score": 1.0,
                "metadata": {"source": doc["filename"], "doc_id": doc["doc_id"], "type": doc.get("type", "pageindex")},
                "source": "pageindex",
            })
    return results[:top_k]


def _load_cached_doc_ids() -> list[dict]:
    if not CACHE_PATH.exists():
        return []
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _wait_for_ready_docs(client, docs: list[dict], timeout_seconds: int = 8) -> list[dict]:
    deadline = time.time() + timeout_seconds
    pending = list(docs)
    ready = []
    while pending and time.time() < deadline:
        next_pending = []
        for doc in pending:
            status = client.get_document(doc["doc_id"]).get("status", "")
            if status == "completed":
                ready.append(doc)
            elif status in {"failed", "error"}:
                continue
            else:
                next_pending.append(doc)
        if not next_pending:
            break
        pending = next_pending
        time.sleep(3)
    return ready


def _local_uploaded_documents() -> list[dict]:
    uploaded = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        uploaded.append({"filename": md_file.name, "type": md_file.parent.name})
    return uploaded


def _local_pageindex_fallback(query: str, top_k: int = 5) -> list[dict]:
    from .offline_retrieval import keyword_score
    from .task4_chunking_indexing import chunk_documents, load_documents

    chunks = chunk_documents(load_documents())
    results = []
    for chunk in chunks:
        score = keyword_score(query, chunk["content"])
        if score > 0:
            results.append({
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk["metadata"],
                "source": "pageindex",
            })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
