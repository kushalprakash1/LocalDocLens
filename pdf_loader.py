from pathlib import Path
import fitz


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract page text from a PDF.
    MVP version: text-based PDFs only.
    Later we add OCR fallback for scanned pages.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")

        pages.append(
            {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "page_number": page_index + 1,
                "text": text.strip(),
            }
        )

    return pages