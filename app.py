"""Streamlit demo UI for the Day 08 group RAG chatbot.

The app first tries the project pipeline from src.task10_generation. If that
pipeline is not ready yet, it falls back to a small local keyword retriever over
the standardized markdown files so the group can still demo the chatbot UI.
"""

from __future__ import annotations

import math
import re
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
PERSONAL_DIR = ROOT_DIR / "personal_submission" / "tuan_anh" / "2A202600758-NguyenTuanAnh-Day08"
CORPUS_DIRS = [
    ROOT_DIR / "data" / "standardized",
    PERSONAL_DIR / "data" / "standardized",
]

SAMPLE_QUESTIONS = [
    "Luật Phòng, chống ma túy 2021 quy định gì về cai nghiện?",
    "Các văn bản pháp luật nào đang được dùng trong hệ thống?",
    "Những tin tức nào liên quan đến nghệ sĩ và ma túy?",
    "Hệ thống có bằng chứng nào cho câu hỏi này không?",
]


st.set_page_config(
    page_title="DrugLaw RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --app-border: #d7dde8;
        --app-muted: #667085;
        --app-accent: #0f766e;
        --app-bg: #f6f8fb;
    }

    .stApp {
        background: linear-gradient(180deg, #f7fafc 0%, #eef3f7 100%);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--app-border);
    }

    .main-header {
        padding: 1.1rem 0 0.35rem;
        border-bottom: 1px solid var(--app-border);
        margin-bottom: 1rem;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 750;
        color: #111827;
        line-height: 1.2;
        margin: 0;
    }

    .main-subtitle {
        color: var(--app-muted);
        font-size: 0.98rem;
        margin-top: 0.35rem;
        max-width: 860px;
    }

    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1rem;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
    }

    .metric-label {
        color: var(--app-muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .source-box {
        background: #ffffff;
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.65rem;
    }

    .source-title {
        color: #111827;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .source-meta {
        color: var(--app-muted);
        font-size: 0.82rem;
        margin-bottom: 0.45rem;
    }

    .source-preview {
        color: #344054;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .small-note {
        color: var(--app-muted);
        font-size: 0.86rem;
        line-height: 1.45;
    }

    @media (max-width: 760px) {
        .metric-row {
            grid-template-columns: 1fr;
        }
        .main-title {
            font-size: 1.55rem;
        }
    }
</style>
"""


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_mode", "none")
    st.session_state.setdefault("pending_question", None)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


@st.cache_data(show_spinner=False)
def load_documents() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for corpus_dir in CORPUS_DIRS:
        if not corpus_dir.exists():
            continue
        for path in sorted(corpus_dir.rglob("*.md")):
            if path.name == ".gitkeep":
                continue
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            relative = path.relative_to(ROOT_DIR)
            docs.append(
                {
                    "content": content,
                    "metadata": {
                        "source": path.stem,
                        "filename": str(relative).replace("\\", "/"),
                        "type": "legal" if "legal" in path.parts else "news",
                        "year": infer_year(path.name, content),
                    },
                    "score": 0.0,
                    "source": "local_demo",
                }
            )
    return docs


def infer_year(filename: str, content: str) -> str:
    for candidate in [filename, content[:500]]:
        match = re.search(r"(20\d{2}|19\d{2})", candidate)
        if match:
            return match.group(1)
    return "n.d."


def split_document(doc: dict[str, Any], chunk_size: int = 900, overlap: int = 140) -> list[dict[str, Any]]:
    text = " ".join(doc["content"].split())
    if not text:
        return []
    chunks = []
    start = 0
    index = 1
    while start < len(text):
        chunk_text = text[start : start + chunk_size]
        metadata = dict(doc["metadata"])
        metadata["chunk"] = index
        chunks.append(
            {
                "content": chunk_text,
                "metadata": metadata,
                "score": 0.0,
                "source": doc.get("source", "local_demo"),
            }
        )
        if start + chunk_size >= len(text):
            break
        start += max(1, chunk_size - overlap)
        index += 1
    return chunks


@st.cache_data(show_spinner=False)
def load_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in load_documents():
        chunks.extend(split_document(doc))
    return chunks


def local_retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    scored = []
    query_set = set(query_terms)
    for chunk in load_chunks():
        content_terms = tokenize(chunk["content"])
        if not content_terms:
            continue
        term_counts = {term: content_terms.count(term) for term in query_set}
        overlap = sum(1 for value in term_counts.values() if value > 0)
        frequency = sum(term_counts.values())
        score = (overlap / max(1, len(query_set))) + math.log1p(frequency) * 0.12
        if score <= 0:
            continue
        result = dict(chunk)
        result["score"] = round(float(score), 4)
        result["source"] = "local_demo"
        scored.append(result)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def source_label(chunk: dict[str, Any], index: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or metadata.get("filename") or f"Nguồn {index}"
    year = metadata.get("year") or "n.d."
    return f"[{source}, {year}]"


def fallback_answer(query: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return (
            "Tôi không thể xác minh thông tin này từ nguồn hiện có. "
            "Hệ thống chưa tìm thấy đoạn tài liệu phù hợp để trích dẫn."
        )

    lines = [
        f"Dựa trên các tài liệu đã truy hồi cho câu hỏi: {query}",
        "",
    ]
    for index, source in enumerate(sources[:3], 1):
        snippet = " ".join(source.get("content", "").split())[:360].rstrip()
        lines.append(f"{index}. {snippet} {source_label(source, index)}")
    lines.append("")
    lines.append(
        "Nếu cần câu trả lời sâu hơn, hãy hỏi tiếp theo hướng cụ thể như điều luật, "
        "nghị định, nhân vật, thời điểm hoặc nguồn báo."
    )
    return "\n".join(lines)


def pipeline_generate(query: str, top_k: int) -> dict[str, Any]:
    try:
        from src.task10_generation import generate_with_citation

        result = generate_with_citation(query, top_k=top_k)
        if isinstance(result, dict) and result.get("answer"):
            result.setdefault("sources", [])
            result.setdefault("retrieval_source", "project_pipeline")
            return result
    except Exception:
        pass

    sources = local_retrieve(query, top_k=top_k)
    return {
        "answer": fallback_answer(query, sources),
        "sources": sources,
        "retrieval_source": "local_demo",
    }


def build_conversation_query(question: str, memory_window: int) -> str:
    if memory_window <= 0:
        return question

    recent = st.session_state.messages[-memory_window * 2 :]
    history_lines = []
    for message in recent:
        role = "Người dùng" if message["role"] == "user" else "Trợ lý"
        content = message["content"].replace("\n", " ")
        history_lines.append(f"{role}: {content[:500]}")

    if not history_lines:
        return question

    return (
        "Ngữ cảnh hội thoại gần đây:\n"
        + "\n".join(history_lines)
        + f"\n\nCâu hỏi mới: {question}"
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("Chưa có nguồn được sử dụng cho câu trả lời này.")
        return

    for index, item in enumerate(sources, 1):
        metadata = item.get("metadata", {}) or {}
        title = escape(str(metadata.get("source") or metadata.get("filename") or f"Nguồn {index}"))
        filename = escape(str(metadata.get("filename", "Không rõ file")))
        doc_type = escape(str(metadata.get("type", "unknown")))
        score = item.get("score", 0)
        preview = escape(" ".join(item.get("content", "").split())[:420])
        st.markdown(
            f"""
            <div class="source-box">
                <div class="source-title">{index}. {title}</div>
                <div class="source-meta">File: {filename} · Loại: {doc_type} · Score: {score}</div>
                <div class="source-preview">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sidebar_controls() -> tuple[int, int]:
    st.sidebar.header("Cấu hình demo")
    top_k = st.sidebar.slider("Số nguồn truy hồi", min_value=3, max_value=8, value=5, step=1)
    memory_window = st.sidebar.slider("Số lượt nhớ hội thoại", min_value=0, max_value=5, value=3, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("Dữ liệu")
    docs = load_documents()
    chunks = load_chunks()
    st.sidebar.write(f"Tài liệu markdown: **{len(docs)}**")
    st.sidebar.write(f"Chunks demo: **{len(chunks)}**")
    st.sidebar.caption("App ưu tiên pipeline chính. Nếu pipeline chưa sẵn sàng, app dùng local retriever để demo.")

    if st.sidebar.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_mode = "none"
        st.rerun()

    return top_k, memory_window


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">DrugLaw RAG Chatbot</h1>
            <div class="main-subtitle">
                Demo chatbot nhóm cho chủ đề pháp luật ma túy và tin tức liên quan.
                Hệ thống truy hồi tài liệu, tạo câu trả lời có citation, ghi nhớ ngữ cảnh hỏi đáp và hiển thị nguồn đã dùng.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    docs = load_documents()
    chunks = load_chunks()
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Tài liệu</div>
                <div class="metric-value">{len(docs)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Chunks</div>
                <div class="metric-value">{len(chunks)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Chế độ</div>
                <div class="metric-value">{st.session_state.last_mode}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sample_questions() -> None:
    st.caption("Câu hỏi gợi ý")
    cols = st.columns(2)
    for index, question in enumerate(SAMPLE_QUESTIONS):
        with cols[index % 2]:
            if st.button(question, key=f"sample_{index}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()


def handle_question(question: str, top_k: int, memory_window: int) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    expanded_query = build_conversation_query(question, memory_window)

    with st.spinner("Đang truy hồi tài liệu và tạo câu trả lời..."):
        result = pipeline_generate(expanded_query, top_k=top_k)

    answer = result.get("answer", "Tôi không thể xác minh thông tin này từ nguồn hiện có.")
    sources = result.get("sources", [])
    mode = result.get("retrieval_source", "unknown")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = sources
    st.session_state.last_mode = mode


def main() -> None:
    init_state()
    top_k, memory_window = sidebar_controls()
    render_header()

    left, right = st.columns([0.64, 0.36], gap="large")

    with left:
        render_sample_questions()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Nhập câu hỏi về pháp luật ma túy hoặc tin tức liên quan...")
        pending = st.session_state.pending_question
        if pending:
            st.session_state.pending_question = None
            handle_question(pending, top_k, memory_window)
            st.rerun()
        if prompt:
            handle_question(prompt, top_k, memory_window)
            st.rerun()

    with right:
        st.subheader("Nguồn đã dùng")
        render_sources(st.session_state.last_sources)
        st.markdown(
            """
            <div class="small-note">
                Citation trong câu trả lời được tạo từ metadata của tài liệu truy hồi.
                Khi có API key, pipeline Task 10 có thể gọi LLM; khi không có, app dùng câu trả lời fallback dựa trên đoạn nguồn.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
