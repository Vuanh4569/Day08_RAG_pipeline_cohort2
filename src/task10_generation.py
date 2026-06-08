from personal_submission.vu_anh.src.task10_generation import (
    generate_with_citation,
    reorder_for_llm,
    format_context
)

if __name__ == "__main__":
    result = generate_with_citation("Cai nghiện ma túy theo luật?")
    print(result["answer"])
