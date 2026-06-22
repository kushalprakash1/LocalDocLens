from pathlib import Path
import fitz


def render_pdf_page_to_image(pdf_path: str | Path, page_index: int, output_dir: str | Path) -> str:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1}.png"
    pix.save(str(image_path))

    return str(image_path)