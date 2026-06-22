import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import lancedb

from localdoc.config import LANCEDB_PATH


def safe_float(value):
    try:
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def load_rows() -> list[dict[str, Any]]:
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table("chunks")
    df = table.to_pandas()
    df["text"] = df["text"].fillna("").astype(str)

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "file_name": str(row["file_name"]),
                "page_number": int(row["page_number"]),
                "text": str(row["text"]),
                "extraction_method": str(row.get("extraction_method", "")),
                "ocr_confidence": safe_float(row.get("ocr_confidence")),
            }
        )

    return rows


def filter_rows(rows: list[dict[str, Any]], file_name: str, page: int = 0) -> list[dict[str, Any]]:
    target = Path(file_name).name.lower()

    selected = [
        row for row in rows
        if Path(row["file_name"]).name.lower() == target or row["file_name"].lower() == file_name.lower()
    ]

    if page > 0:
        selected = [row for row in selected if row["page_number"] == page]

    if not selected:
        available = sorted({row["file_name"] for row in rows})
        file_list = "\n".join(f"- {name}" for name in available)

        raise RuntimeError(
            f"No extracted text found for file: {file_name}\n\n"
            f"Available indexed files:\n{file_list}\n\n"
            "Run localdoc ingest data/docs after copying your PDF into data/docs."
        )

    return selected


def extract_file_text(file_name: str, page: int = 0, output_dir: str = "artifacts") -> dict[str, Any]:
    rows = load_rows()
    selected = filter_rows(rows, file_name=file_name, page=page)

    selected = sorted(selected, key=lambda row: (row["page_number"], row["chunk_id"]))

    pages = defaultdict(list)

    for row in selected:
        pages[row["page_number"]].append(row)

    page_outputs = []

    for page_number in sorted(pages):
        page_rows = pages[page_number]

        page_text = "\n\n".join(row["text"] for row in page_rows)

        methods = sorted({row["extraction_method"] for row in page_rows if row["extraction_method"]})
        confidences = [
            row["ocr_confidence"]
            for row in page_rows
            if row["ocr_confidence"] is not None
        ]

        avg_ocr_confidence = None

        if confidences:
            avg_ocr_confidence = round(sum(confidences) / len(confidences), 4)

        page_outputs.append(
            {
                "page_number": page_number,
                "chunks": len(page_rows),
                "extraction_methods": methods,
                "avg_ocr_confidence": avg_ocr_confidence,
                "text": page_text,
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(file_name).stem.replace(" ", "_")

    if page > 0:
        txt_path = output_path / f"{safe_stem}_page_{page}_extracted_text.txt"
        json_path = output_path / f"{safe_stem}_page_{page}_extracted_text.json"
    else:
        txt_path = output_path / f"{safe_stem}_extracted_text.txt"
        json_path = output_path / f"{safe_stem}_extracted_text.json"

    text_lines = []

    text_lines.append(f"Extracted text for: {file_name}")
    text_lines.append(f"Pages extracted: {[item['page_number'] for item in page_outputs]}")
    text_lines.append("")

    for item in page_outputs:
        text_lines.append("=" * 80)
        text_lines.append(f"PAGE {item['page_number']}")
        text_lines.append(f"Chunks: {item['chunks']}")
        text_lines.append(f"Extraction methods: {item['extraction_methods']}")
        text_lines.append(f"Average OCR confidence: {item['avg_ocr_confidence']}")
        text_lines.append("=" * 80)
        text_lines.append("")
        text_lines.append(item["text"])
        text_lines.append("")

    result = {
        "file_name": file_name,
        "page_filter": page,
        "pages": page_outputs,
        "output_files": {
            "text": str(txt_path),
            "json": str(json_path),
        },
    }

    txt_path.write_text("\n".join(text_lines), encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    result = extract_file_text(file_name="scanned_supplier_packet.pdf")
    print(json.dumps(result, indent=2))
