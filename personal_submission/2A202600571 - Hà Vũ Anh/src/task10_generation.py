"""
Task 10 — Generation Có Citation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

# Configuration
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021] or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.
    Pattern: [1, 3, 5, 4, 2] từ input [1, 2, 3, 4, 5]
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    # Odd indices first (0, 2, 4...)
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    # Even indices reversed (..., 3, 1)
    start_even = len(chunks) - 1
    if start_even % 2 != 0:
         # odd index for last element, we need the largest even index
         pass
    else:
         start_even -= 1
    for i in range(start_even, 0, -2):
        reordered.append(chunks[i])
        
    return reordered


def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string cho prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation có citation."""
    chunks = retrieve(query, top_k=top_k)
    
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none"
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    # Check if a valid OpenAI API key is set
    if openai_key and not openai_key.startswith("sk-xxx"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": chunks,
                "retrieval_source": chunks[0].get("source", "hybrid")
            }
        except Exception as e:
            print(f"⚠ OpenAI LLM call failed: {e}. Falling back to mock generator.")
            
    # Mock RAG generator fallback (to make testing completely robust offline)
    print("Running local mock RAG generator fallback...")
    
    # Generate simple answer referencing the sources
    paragraphs = []
    for chunk in chunks[:3]:
        source = chunk.get("metadata", {}).get("source", "Tài liệu")
        # clean source name for citation
        source_clean = source.replace(".md", "").replace(".pdf", "").replace("-", " ").title()
        
        # Simple extraction
        sentence = chunk["content"].split("\n")[0]
        if len(sentence) < 30 and len(chunk["content"]) > 30:
            sentence = chunk["content"][:100] + "..."
            
        paragraphs.append(f"Theo nguồn {source_clean}, {sentence} [{source_clean}].")
        
    answer = " ".join(paragraphs)
    
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid")
    }


if __name__ == "__main__":
    q = "Chi Dân bị bắt vì sử dụng ma tuý như thế nào?"
    res = generate_with_citation(q)
    print(f"\nQuery: {q}")
    print(f"Answer:\n{res['answer']}")
    print(f"Retrieval source: {res['retrieval_source']}")
