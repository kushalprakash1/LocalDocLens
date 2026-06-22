from localdoc.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def make_page_chunks(page: dict) -> list[dict]:
    chunks = chunk_text(page["text"])

    records = []

    for idx, chunk in enumerate(chunks):
        records.append(
            {
                "chunk_id": f"{page['file_name']}::p{page['page_number']}::c{idx}",
        	"file_name": page["file_name"],
        	"file_path": page["file_path"],
        	"page_number": page["page_number"],
        	"chunk_index": idx,
        	"text": chunk,
        	"level": "page_chunk",
        	"chunk_type": "text",
        	"bbox": None,
        	"ocr_confidence": page.get("ocr_confidence"),
        	"extraction_method": page.get("extraction_method", "pdf_text"),
            }
        )

    return records
