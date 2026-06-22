from pathlib import Path
import fitz

INPUT_PDF = Path("data/docs/sample_supplier_packet.pdf")
OUTPUT_PDF = Path("data/docs/scanned_supplier_packet.pdf")
TEMP_DIR = Path("data/scanned_pages")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

src = fitz.open(INPUT_PDF)
out = fitz.open()

for i, page in enumerate(src):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_path = TEMP_DIR / f"page_{i + 1}.png"
    pix.save(str(img_path))

    rect = page.rect
    new_page = out.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, filename=str(img_path))

out.save(OUTPUT_PDF)
out.close()

print(f"Created image-only scanned PDF: {OUTPUT_PDF}")