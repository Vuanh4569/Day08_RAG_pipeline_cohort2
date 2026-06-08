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
import ssl
import urllib.error
import urllib.request

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCS = [
    {
        "url": "https://congan.sonla.gov.vn/wp-content/uploads/2022/05/1.-Luat-PCMT-2021.pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
        "filename": "nghi-dinh-105-2021.pdf",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/08/57-cp.signed.pdf",
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Download a legal document to data/landing/legal/."""
    setup_directory()
    filepath = DATA_DIR / filename
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise RuntimeError(f"Không tải được {url}: {exc}") from exc
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=90, context=context) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Không tải được {url}: {exc}") from exc

    if len(content) < 1024:
        raise RuntimeError(f"Downloaded file too small: {url}")
    filepath.write_bytes(content)
    print(f"Đã tải: {filepath} ({len(content)} bytes)")
    return filepath


def download_all() -> list[Path]:
    """Download the minimum 3 legal files required by Task 1."""
    downloaded = []
    for doc in LEGAL_DOCS:
        downloaded.append(download_file(doc["url"], doc["filename"]))
    return downloaded


if __name__ == "__main__":
    download_all()
