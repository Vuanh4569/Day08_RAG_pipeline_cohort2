import sys
import unittest
from pathlib import Path


GROUP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GROUP_DIR))


class TestRagAdapter(unittest.TestCase):
    def test_empty_question_returns_prompt_message(self):
        from rag_adapter import answer_question

        result = answer_question("   ")

        self.assertIn("nhập câu hỏi", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieval_source"], "none")

    def test_format_sources_markdown_includes_metadata(self):
        from rag_adapter import format_sources_markdown

        markdown = format_sources_markdown(
            [
                {
                    "content": "Nội dung điều luật về phòng chống ma túy.",
                    "score": 0.87,
                    "source": "hybrid",
                    "metadata": {
                        "source": "luat-phong-chong-ma-tuy-2021.md",
                        "type": "legal",
                    },
                }
            ]
        )

        self.assertIn("luat-phong-chong-ma-tuy-2021.md", markdown)
        self.assertIn("legal", markdown)
        self.assertIn("0.8700", markdown)


if __name__ == "__main__":
    unittest.main()
