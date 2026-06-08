import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


class TestRetrievalQuality(unittest.TestCase):
    def test_clean_markdown_removes_pdf_signature_noise(self):
        from src.task4_chunking_indexing import clean_markdown_text

        noisy = "Intro\n<</Type/Sig/Filter/Adobe.PPKLite/SubFilter/adbe.pkcs7.detached/Reason(foo)>>\nĐiều 1. Nội dung"
        cleaned = clean_markdown_text(noisy)

        self.assertNotIn("Adobe.PPKLite", cleaned)
        self.assertIn("Điều 1. Nội dung", cleaned)

    def test_cai_nghien_query_retrieves_article_28(self):
        from src.task9_retrieval_pipeline import retrieve

        results = retrieve(
            "Luật Phòng chống ma túy 2021 quy định những hình thức cai nghiện nào?",
            top_k=5,
        )
        joined = "\n".join(item["content"] for item in results)

        self.assertIn("Điều 28. Các biện pháp cai nghiện ma túy", joined)
        self.assertIn("Cai nghiện ma túy tự nguyện", joined)
        self.assertIn("Cai nghiện ma túy bắt buộc", joined)


if __name__ == "__main__":
    unittest.main()
