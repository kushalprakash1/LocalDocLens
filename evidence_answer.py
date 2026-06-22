def format_evidence_answer(query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant evidence found."

    lines = []
    lines.append(f"Query: {query}")
    lines.append("")
    lines.append("Top evidence:")
    lines.append("")

    for i, r in enumerate(results, start=1):
        text = r["text"]
        preview = text[:500] + ("..." if len(text) > 500 else "")

        lines.append(f"{i}. {r['file_name']} — page {r['page_number']}")
        lines.append(f"   Chunk ID: {r['chunk_id']}")
        lines.append(f"   Score: {r.get('hybrid_score', 'n/a')}")
        lines.append(f"   Evidence: {preview}")
        lines.append("")

    return "\n".join(lines)