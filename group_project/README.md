# Bài Tập Nhóm — RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm xây dựng **RAG Chatbot** trả lời câu hỏi về pháp luật ma túy và tin tức liên quan, có citation và hiển thị source documents.

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Nhóm triển khai Gradio chatbot trong `group_project/app.py`.

**Tính năng đã triển khai:**
- Giao diện chat bằng Gradio.
- Trả lời có citation dựa trên metadata của source chunks.
- Hỗ trợ follow-up questions bằng conversation memory ngắn.
- Hiển thị source documents đã dùng, gồm source, type, score, retrieval mode và content preview.
- Có controls cho `top_k`, bật/tắt memory và bật/tắt reranking.

**Stack gợi ý:**
```
Gradio → rag_adapter.py → Retrieval (Task 9) → Generation (Task 10) → Display answer + sources
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
Raw legal/news data
  ↓
personal_submission/<folder cá nhân>/data/landing
  ↓
Task 3 convert markdown
  ↓
personal_submission/<folder cá nhân>/data/standardized
  ↓
Task 4 chunking/indexing
  ↓
Task 5 semantic search + Task 6 lexical search
  ↓
Task 9 hybrid retrieval + reranking + PageIndex fallback
  ↓
Task 10 generation with citation
  ↓
group_project/rag_adapter.py
  ↓
group_project/app.py Gradio Chat UI
```

---

## Data Cho Demo

Hiện demo dùng data đã xử lý trong `personal_submission/<folder cá nhân>/data/standardized/`:

- `legal/`: văn bản pháp luật về ma túy.
- `news/`: bài báo đã crawl về nghệ sĩ liên quan đến ma túy.

Khi có data của các thành viên khác, nhóm có thể merge thêm vào corpus bằng cách copy markdown đã chuẩn hóa vào cùng cấu trúc `legal/` hoặc `news/`, sau đó chạy lại indexing/retrieval ở phần cá nhân.

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Phạm Đình Phúc | 2A202600802 | Pipeline cá nhân, Gradio chatbot adapter/UI, source display | Done |
| | | | |
| | | | |
| | | | |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r "personal_submission/2A202600802 - Phạm Đình Phúc/requirements.txt"
pip install -r group_project/requirements.txt

# Chạy app
python group_project/app.py
```

Smoke test adapter:

```bash
cd group_project
python -c "from rag_adapter import answer_question; print(answer_question('Luật phòng chống ma túy quy định gì?', top_k=2, use_reranking=False)['answer'])"
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
