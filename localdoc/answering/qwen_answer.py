import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:4b"


def build_context(results: list[dict]) -> str:
    blocks = []

    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"File: {r['file_name']}\n"
            f"Page: {r['page_number']}\n"
            f"Chunk ID: {r['chunk_id']}\n"
            f"Evidence text: {r['text']}\n"
        )

    return "\n\n".join(blocks)


def answer_with_qwen(query: str, results: list[dict], model: str = DEFAULT_MODEL) -> str:
    if not results:
        return "No relevant evidence found."

    context = build_context(results)

    prompt = f"""
You are LocalDocLens, a local supplier-document RAG assistant.

Answer the user's question using ONLY the evidence below.
Do not use outside knowledge.
If the evidence is not enough, say that the evidence is not enough.
Always mention the file name and page number for each important claim.

User question:
{query}

Evidence:
{context}

Write a clear answer with:
1. Direct answer
2. Supporting evidence
3. Any uncertainty or missing information
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()
