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
import re
import time
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
PAGEINDEX_MANIFEST = INDEX_DIR / "pageindex_manifest.json"
PAGEINDEX_DOC_IDS = INDEX_DIR / "pageindex_doc_ids.json"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _load_markdown_documents() -> list[dict]:
    """Load markdown documents for the local vectorless fallback."""
    docs = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = md_file.parent.name
        docs.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
    return docs


def _lexical_score(query_tokens: list[str], content: str) -> float:
    """
    Vectorless score: BM25-like term matching over document text.

    PageIndex thật dùng cấu trúc tài liệu thay vì vector embedding. Bản fallback
    local này cũng không dùng vector; nó xếp hạng bằng từ khóa và metadata.
    """
    doc_tokens = _tokenize(content)
    if not query_tokens or not doc_tokens:
        return 0.0

    freqs = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        tf = freqs.get(token, 0)
        if tf:
            score += (tf * 2.5) / (tf + 1.5 + 0.75 * doc_len / 500)

    return score


def upload_documents():
    """
    Upload các PDF luật gốc lên PageIndex Cloud bằng Python SDK.

    PageIndex Document Processing hiện nhận PDF, nên Task 8 upload 3 file
    pháp luật trong data/landing/legal thay vì các file markdown đã convert.
    Hàm lưu doc_id vào data/index/pageindex_doc_ids.json để pageindex_search()
    có thể query lại các tài liệu đã xử lý.
    """
    pdf_files = sorted(
        pdf_file for pdf_file in LEGAL_DIR.glob("*.pdf")
        if ".original" not in pdf_file.name
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        return _write_local_manifest("missing-api-key")

    try:
        from pageindex import PageIndexClient
    except Exception as exc:
        return _write_local_manifest(f"sdk-import-failed: {exc}")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    existing = _load_pageindex_doc_ids()
    uploaded = []

    for pdf_file in pdf_files:
        current = next((item for item in existing if item.get("filename") == pdf_file.name), None)
        if current and current.get("doc_id"):
            print(f"Exists on PageIndex, skip upload: {pdf_file.name}")
            uploaded.append(current)
            continue

        print(f"Uploading to PageIndex: {pdf_file.name}")
        try:
            result = client.submit_document(str(pdf_file))
            uploaded.append({
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "doc_id": result["doc_id"],
            })
        except Exception as exc:
            print(f"PageIndex upload failed for {pdf_file.name}: {exc}")

    if uploaded:
        PAGEINDEX_DOC_IDS.write_text(
            json.dumps(uploaded, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved PageIndex doc ids: {PAGEINDEX_DOC_IDS}")
        return uploaded

    return _write_local_manifest("pageindex-upload-failed")


def _write_local_manifest(reason: str) -> Path:
    """Prepare fallback manifest when PageIndex Cloud cannot be used."""
    docs = _load_markdown_documents()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST.write_text(
        json.dumps(
            {
                "mode": "local-vectorless-fallback",
                "reason": reason,
                "pageindex_api_configured": bool(PAGEINDEX_API_KEY and not PAGEINDEX_API_KEY.startswith("pi_xxx")),
                "documents": [
                    {"metadata": doc["metadata"], "chars": len(doc["content"])}
                    for doc in docs
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(docs)} documents for local PageIndex fallback ({reason})")
    return PAGEINDEX_MANIFEST


def _load_pageindex_doc_ids() -> list[dict]:
    if not PAGEINDEX_DOC_IDS.exists():
        return []
    return json.loads(PAGEINDEX_DOC_IDS.read_text(encoding="utf-8"))


def _local_pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Local vectorless fallback used when PageIndex Cloud is unavailable."""
    docs = _load_markdown_documents()
    query_tokens = _tokenize(query)

    ranked = []
    for doc in docs:
        score = _lexical_score(query_tokens, doc["content"])
        ranked.append({
            "content": doc["content"],
            "score": float(score),
            "metadata": {
                **doc["metadata"],
                "retrieval_mode": "local-vectorless-pageindex-fallback",
            },
            "source": "pageindex",
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _extract_retrieved_nodes(retrieval_result: dict, filename: str, doc_id: str) -> list[dict]:
    results = []
    for node in retrieval_result.get("retrieved_nodes", []) or []:
        title = node.get("title", "")
        for item in node.get("relevant_contents", []) or []:
            content = item.get("relevant_content") or item.get("text") or ""
            if not content:
                continue
            results.append({
                "content": content,
                "score": 1.0,
                "metadata": {
                    "source": filename,
                    "doc_id": doc_id,
                    "node_id": node.get("node_id"),
                    "title": title,
                    "page_index": item.get("page_index"),
                    "retrieval_mode": "pageindex-cloud",
                },
                "source": "pageindex",
            })
    return results


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
    if top_k <= 0:
        return []

    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        return _local_pageindex_search(query, top_k)

    try:
        from pageindex import PageIndexClient

        doc_infos = _load_pageindex_doc_ids()
        if not doc_infos:
            if PAGEINDEX_MANIFEST.exists():
                return _local_pageindex_search(query, top_k)
            upload_documents()
            doc_infos = _load_pageindex_doc_ids()

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        all_results = []

        for doc in doc_infos:
            doc_id = doc["doc_id"]
            filename = doc.get("filename", doc_id)
            document = client.get_document(doc_id)
            if document.get("status") != "completed":
                print(f"PageIndex document not ready yet: {filename} ({document.get('status')})")
                continue

            if not client.is_retrieval_ready(doc_id):
                print(f"PageIndex retrieval not ready yet: {filename}")
                continue

            retrieval = client.submit_query(doc_id=doc_id, query=query, thinking=False)
            retrieval_id = retrieval["retrieval_id"]

            retrieval_result = {}
            for _ in range(6):
                retrieval_result = client.get_retrieval(retrieval_id)
                if retrieval_result.get("status") == "completed":
                    break
                time.sleep(2)

            if retrieval_result.get("status") == "completed":
                all_results.extend(_extract_retrieved_nodes(retrieval_result, filename, doc_id))

        if all_results:
            return all_results[:top_k]

    except Exception as exc:
        print(f"PageIndex Cloud query failed, using local fallback: {exc}")

    return _local_pageindex_search(query, top_k)


if __name__ == "__main__":
    upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        preview = r["content"][:100].encode("ascii", errors="ignore").decode("ascii")
        print(f"[{r['score']:.3f}] {preview}...")
