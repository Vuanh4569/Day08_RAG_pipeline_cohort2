"""Gradio RAG chatbot for the Day08 group project."""

from __future__ import annotations

import os

import gradio as gr

from rag_adapter import answer_question


EXAMPLES = [
    "Luật Phòng chống ma túy 2021 quy định những hình thức cai nghiện nào?",
    "Nghệ sĩ nào trong dữ liệu báo chí liên quan đến ma túy?",
    "Nếu hybrid search không đủ tốt thì pipeline fallback thế nào?",
]


def respond(
    message: str,
    history: list[tuple[str, str]],
    top_k: int,
    use_memory: bool,
    use_reranking: bool,
):
    result = answer_question(
        message,
        history=history,
        top_k=int(top_k),
        use_memory=use_memory,
        use_reranking=use_reranking,
    )
    history = history + [(message, result["answer"])]
    return "", history, result["source_markdown"]


def clear_chat():
    return [], "_Chưa có nguồn nào được dùng._"


with gr.Blocks(title="DrugLaw RAG Chatbot") as demo:
    gr.Markdown(
        "# DrugLaw RAG Chatbot\n"
        "Chatbot trả lời câu hỏi về pháp luật ma túy và tin tức liên quan, kèm citation và source chunks."
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
                show_copy_button=True,
            )
            message = gr.Textbox(
                label="Câu hỏi",
                placeholder="Nhập câu hỏi về pháp luật ma túy hoặc tin tức liên quan...",
                lines=2,
            )
            with gr.Row():
                submit = gr.Button("Gửi", variant="primary")
                clear = gr.Button("Xóa chat")

            gr.Examples(EXAMPLES, inputs=message)

        with gr.Column(scale=1):
            top_k = gr.Slider(1, 8, value=5, step=1, label="Top K nguồn")
            use_memory = gr.Checkbox(value=True, label="Dùng conversation memory")
            use_reranking = gr.Checkbox(value=True, label="Dùng reranking")
            sources = gr.Markdown("_Chưa có nguồn nào được dùng._", label="Sources")

    submit.click(
        respond,
        inputs=[message, chatbot, top_k, use_memory, use_reranking],
        outputs=[message, chatbot, sources],
    )
    message.submit(
        respond,
        inputs=[message, chatbot, top_k, use_memory, use_reranking],
        outputs=[message, chatbot, sources],
    )
    clear.click(clear_chat, outputs=[chatbot, sources])


if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=server_port)
