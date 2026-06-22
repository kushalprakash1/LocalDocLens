from pathlib import Path
import fitz

from localdoc.config import RENDERED_PAGES_DIR, MIN_TEXT_CHARS_BEFORE_OCR
from localdoc.ingestion.page_renderer import render_pdf_page_to_image


def load_pdf_pages(pdf_path: str | Path, use_ocr: bool = True) -> list[dict]:
    """
    Extract page text from a PDF.
    If a page has little/no selectable text, render it and run OCR.

    Important:
    OCR is lazy-loaded only when needed so PaddleOCR does not load at CLI startup.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []
    ocr_engine = None

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()

        ocr_confidence = None
        extraction_method = "pdf_text"
        ocr_boxes = []

        if use_ocr and len(text) < MIN_TEXT_CHARS_BEFORE_OCR:
            # Lazy import PaddleOCR only when this page actually needs OCR.
            if ocr_engine is None:
                from localdoc.ocr.printed_ocr import PrintedOCR
                ocr_engine = PrintedOCR()

            image_path = render_pdf_page_to_image(
                pdf_path=pdf_path,
                page_index=page_index,
                output_dir=RENDERED_PAGES_DIR,
            )

            ocr_result = ocr_engine.extract_text(image_path)
            text = ocr_result["text"].strip()
            ocr_confidence = ocr_result["confidence"]
            ocr_boxes = ocr_result["boxes"]
            extraction_method = "ocr"

        pages.append(
            {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "page_number": page_index + 1,
                "text": text,
                "ocr_confidence": ocr_confidence,
                "ocr_boxes": ocr_boxes,
                "extraction_method": extraction_method,
            }
        )

    return pages