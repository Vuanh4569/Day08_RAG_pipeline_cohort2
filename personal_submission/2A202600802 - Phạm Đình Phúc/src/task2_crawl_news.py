"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Tavily Search API để tìm và lấy nội dung bài báo.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Yêu cầu:
    TAVILY_API_KEY trong .env
"""

import asyncio
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .env_utils import get_env

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Curated URLs đúng chủ đề; Tavily Extract sẽ lấy nội dung từ các URL này.
ARTICLE_URLS = [
    "https://dantri.com.vn/phap-luat/truoc-ca-si-chu-bin-loat-nghe-si-noi-tieng-vuong-lao-ly-vi-ma-tuy-20240608123002810.htm",
    "https://vnexpress.net/nguoi-mau-nhikolai-dinh-bi-bat-vi-tang-tru-ma-tuy-4762598.html",
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-bi-khoi-to-toi-giet-nguoi-3840141.html",
    "https://dantri.com.vn/phap-luat/sau-lum-xum-dinh-ma-tuy-rapper-binh-gold-bi-bat-vi-cuop-tai-san-20250726152734511.htm",
]

DEFAULT_QUERIES = [
    "nghệ sĩ Việt Nam bị bắt ma túy",
    "ca sĩ Việt Nam liên quan ma túy",
    "diễn viên Việt Nam sử dụng ma túy",
    "người nổi tiếng Việt Nam ma túy bị xử lý",
    "site:vnexpress.net nghệ sĩ ma túy",
    "site:tuoitre.vn nghệ sĩ ma túy",
    "site:thanhnien.vn nghệ sĩ ma túy",
    "site:dantri.com.vn nghệ sĩ ma túy",
]


def _slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug[:80] or fallback).strip("-")


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Tavily request failed: HTTP {exc.code} {detail}") from exc


def tavily_search_articles(max_results: int = 5) -> list[dict]:
    """Search Tavily and return normalized article records with markdown content."""
    api_key = get_env("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    articles = []
    seen_urls = set()
    for query in DEFAULT_QUERIES:
        payload = {
            "query": query,
            "topic": "news",
            "search_depth": "advanced",
            "max_results": max(max_results, 10),
            "include_raw_content": "markdown",
            "include_answer": False,
            "include_images": False,
        }
        response = _post_json("https://api.tavily.com/search", payload, {"Authorization": f"Bearer {api_key}"})
        for result in response.get("results", []):
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            raw_content = result.get("raw_content") or result.get("content") or ""
            if len(raw_content.strip()) < 250:
                continue
            seen_urls.add(url)
            articles.append({
                "url": url,
                "title": result.get("title") or "Untitled article",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": raw_content,
                "query": query,
                "score": result.get("score", 0),
            })
            if len(articles) >= max_results:
                return articles
    return articles


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    api_key = get_env("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in .env")

    payload = {"urls": [url], "extract_depth": "advanced", "format": "markdown"}
    response = _post_json("https://api.tavily.com/extract", payload, {"Authorization": f"Bearer {api_key}"})
    result = (response.get("results") or [{}])[0]
    return {
        "url": url,
        "title": result.get("title") or url,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": result.get("raw_content") or result.get("content") or "",
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    if ARTICLE_URLS:
        articles = []
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            articles.append(await crawl_article(url))
    else:
        print("Searching Tavily for Vietnamese celebrity drug-related news...")
        articles = tavily_search_articles(max_results=5)

    if len(articles) < 5:
        raise RuntimeError(f"Need at least 5 articles, got {len(articles)}")

    for i, article in enumerate(articles[:5], 1):
        filename = f"{i:02d}-{_slugify(article.get('title', ''), f'article-{i:02d}')}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
