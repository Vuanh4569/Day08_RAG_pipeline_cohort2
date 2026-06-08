"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"

            try:
                result = md.convert(str(filepath))
                text = (result.text_content or "").strip()
            except Exception as exc:
                # Keep the pipeline runnable even when a PDF parser dependency
                # is missing. Later retrieval still has source metadata.
                text = (
                    f"Không thể trích xuất toàn văn tự động từ file {filepath.name}.\n\n"
                    f"Lỗi MarkItDown: {exc}\n\n"
                    "Tài liệu này là văn bản pháp luật gốc được lưu trong "
                    "data/landing/legal để phục vụ RAG pipeline về phòng, "
                    "chống ma túy."
                )

            header = (
                f"# {filepath.stem}\n\n"
                f"**Source file:** {filepath.name}\n"
                f"**Type:** legal\n\n"
                "---\n\n"
            )
            output_path.write_text(header + text + "\n", encoding="utf-8")
            converted.append(output_path)
            print(f"  Saved: {output_path}")

    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            title = data.get("title", filepath.stem)
            source = data.get("source", "N/A")
            url = data.get("url", "N/A")
            crawl_date = data.get("crawl_date") or data.get("date_crawled", "N/A")
            published_at = data.get("published_at", "N/A")
            topic = data.get("topic", "news")
            body = (
                data.get("content_markdown")
                or data.get("markdown")
                or data.get("content")
                or data.get("text")
                or ""
            )

            metadata = data.get("metadata", {}) or {}
            metadata_lines = "\n".join(
                f"- **{key}:** {value}" for key, value in metadata.items()
            )
            if metadata_lines:
                metadata_lines = f"\n## Metadata\n\n{metadata_lines}\n"

            content = (
                f"# {title}\n\n"
                f"**Source:** {source}\n"
                f"**URL:** {url}\n"
                f"**Published:** {published_at}\n"
                f"**Crawled:** {crawl_date}\n"
                f"**Type:** news\n"
                f"**Topic:** {topic}\n\n"
                "---\n\n"
                f"{body}\n"
                f"{metadata_lines}"
            )
            output_path.write_text(content, encoding="utf-8")
            converted.append(output_path)
            print(f"  Saved: {output_path}")

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    print(f"\nDone! Converted {len(legal_files) + len(news_files)} files.")
    print("Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
