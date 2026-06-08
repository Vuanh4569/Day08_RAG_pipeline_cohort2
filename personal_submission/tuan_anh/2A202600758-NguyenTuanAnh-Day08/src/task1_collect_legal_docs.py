"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


LEGAL_DOCUMENTS = [
    {
        "title": "Luật Phòng, chống ma túy 2021",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "url": "https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbRjbik8E0wBHUyBAiSflcQEJHAmcZL6yTIQL6RZmX61EtZRIA6qhCdHI7bAwaisGDmaN_1mqn-kBerf_4AVrsSdg~~&file_name=2021_567+%2B+568_73-2021-QH14.pdf",
        "source": "Công báo Chính phủ",
    },
    {
        "title": "Nghị định 105/2021/NĐ-CP",
        "filename": "nghi-dinh-105-2021.pdf",
        "url": "https://congbao.chinhphu.vn/tai-ve-van-ban-so-105-2021-nd-cp-34944-37821?format=pdf",
        "source": "Công báo Chính phủ",
    },
    {
        "title": "Nghị định 116/2021/NĐ-CP",
        "filename": "nghi-dinh-116-2021.pdf",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/116.signed.pdf",
        "source": "Cổng Thông tin điện tử Chính phủ",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Download one legal PDF/DOCX into data/landing/legal/."""
    filepath = DATA_DIR / filename
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request, timeout=60) as response:
        content = response.read()

    if len(content) <= 1024:
        raise ValueError(f"Downloaded file is too small: {filename}")

    filepath.write_bytes(content)
    print(f"Downloaded: {filepath}")
    return filepath


def collect_legal_docs(force: bool = False) -> list[Path]:
    """
    Download the three required legal documents for Task 1.

    Args:
        force: Redownload files even if they already exist.

    Returns:
        List of local file paths.
    """
    setup_directory()
    downloaded = []

    for doc in LEGAL_DOCUMENTS:
        filepath = DATA_DIR / doc["filename"]
        if filepath.exists() and filepath.stat().st_size > 1024 and not force:
            print(f"Exists, skip: {filepath.name}")
            downloaded.append(filepath)
            continue

        print(f"Downloading {doc['title']} from {doc['source']}...")
        downloaded.append(download_file(doc["url"], doc["filename"]))

    return downloaded


if __name__ == "__main__":
    collect_legal_docs()
