"""Recrawl legal markdown with Tavily to avoid noisy signed-PDF conversion."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .env_utils import get_env


OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized" / "legal"

LEGAL_QUERIES = [
    {
        "filename": "nghi-dinh-105-2021.md",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
        "query": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng chống ma túy nội dung",
    },
    {
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.md",
        "title": "Nghị định 57/2022/NĐ-CP danh mục chất ma túy và tiền chất",
        "query": "Nghị định 57/2022/NĐ-CP danh mục chất ma túy tiền chất nội dung",
    },
]


def recrawl_all() -> list[Path]:
    api_key = get_env("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for doc in LEGAL_QUERIES:
        result = _search_best_result(api_key, doc["query"])
        content = _extract_url(api_key, result["url"]) or result.get("content", "")
        content = _clean_tavily_content(content)
        if len(content) < 500:
            raise RuntimeError(f"Tavily content too short for {doc['title']}: {result['url']}")

        path = OUTPUT_DIR / doc["filename"]
        markdown = _format_markdown(doc["title"], result, content)
        path.write_text(markdown, encoding="utf-8")
        written.append(path)
    return written


def _search_best_result(api_key: str, query: str) -> dict:
    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": 8,
        "include_raw_content": "markdown",
        "include_answer": False,
        "include_images": False,
    }
    response = _post_json("https://api.tavily.com/search", payload, api_key)
    results = response.get("results", [])
    if not results:
        raise RuntimeError(f"Tavily returned no results for query: {query}")

    preferred_domains = ("vanban.chinhphu.vn", "thuvienphapluat.vn", "luatvietnam.vn")
    for result in results:
        url = result.get("url", "")
        if any(domain in url for domain in preferred_domains):
            return result
    return results[0]


def _extract_url(api_key: str, url: str) -> str:
    payload = {"urls": [url], "extract_depth": "advanced", "format": "markdown"}
    response = _post_json("https://api.tavily.com/extract", payload, api_key)
    result = (response.get("results") or [{}])[0]
    return result.get("raw_content") or result.get("content") or ""


def _post_json(url: str, payload: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Tavily request failed: HTTP {exc.code} {detail}") from exc


def _clean_tavily_content(content: str) -> str:
    content = content.replace("\x00", "")
    content = re.sub(r"<</Type/Sig.*?>>", "", content, flags=re.DOTALL)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _format_markdown(title: str, result: dict, content: str) -> str:
    url = result.get("url", "N/A")
    result_title = result.get("title") or title
    crawled_at = datetime.now().isoformat()
    return (
        f"# {title}\n\n"
        f"**Title:** {result_title}\n"
        f"**Source:** {url}\n"
        f"**URL:** {url}\n"
        f"**Crawled:** {crawled_at}\n\n"
        "---\n\n"
        f"{content}\n"
    )


if __name__ == "__main__":
    for output_path in recrawl_all():
        print(f"Saved: {output_path}")
