import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import lancedb

from localdoc.config import LANCEDB_PATH


CACHE_DB_PATH = Path("data/db/answer_cache.sqlite")


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


def load_cache_stats() -> dict[str, Any]:
    if not CACHE_DB_PATH.exists():
        return {
            "cache_db_path": str(CACHE_DB_PATH),
            "exists": False,
            "total_cached_answers": 0,
            "verified_cached_answers": 0,
            "total_cache_hits": 0,
        }

    with sqlite3.connect(CACHE_DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM answer_cache WHERE verified = 1").fetchone()[0]
        hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM answer_cache").fetchone()[0]

    return {
        "cache_db_path": str(CACHE_DB_PATH),
        "exists": True,
        "total_cached_answers": int(total),
        "verified_cached_answers": int(verified),
        "total_cache_hits": int(hits),
    }


def inspect_index(output_dir: str = "artifacts") -> dict[str, Any]:
    rows = load_rows()

    by_file = defaultdict(list)

    for row in rows:
        by_file[row["file_name"]].append(row)

    files = []

    for file_name, file_rows in sorted(by_file.items()):
        pages = sorted({row["page_number"] for row in file_rows})
        methods = sorted({row["extraction_method"] for row in file_rows if row["extraction_method"]})
        confidences = [
            row["ocr_confidence"]
            for row in file_rows
            if row["ocr_confidence"] is not None
        ]

        avg_ocr_confidence = None

        if confidences:
            avg_ocr_confidence = round(sum(confidences) / len(confidences), 4)

        files.append(
            {
                "file_name": file_name,
                "chunks": len(file_rows),
                "pages": pages,
                "page_count": len(pages),
                "extraction_methods": methods,
                "avg_ocr_confidence": avg_ocr_confidence,
                "sample_chunk_ids": [row["chunk_id"] for row in file_rows[:5]],
            }
        )

    result = {
        "total_files": len(files),
        "total_chunks": len(rows),
        "files": files,
        "cache": load_cache_stats(),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "index_inspection.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["output_file"] = str(json_path)

    return result


if __name__ == "__main__":
    print(json.dumps(inspect_index(), indent=2))
