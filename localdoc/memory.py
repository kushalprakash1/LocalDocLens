import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from localdoc.facts import extract_facts_for_file


MEMORY_DB_PATH = Path("data/db/supplier_memory.sqlite")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_file_name(file_name: str) -> str:
    return Path(file_name).name.lower().strip()


def compute_file_fingerprint(rows: list[dict[str, Any]]) -> str:
    stable_rows = []

    for row in sorted(rows, key=lambda item: (item["file_name"], item["page_number"], item["chunk_id"])):
        stable_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "file_name": row["file_name"],
                "page_number": row["page_number"],
                "text": row["text"],
                "extraction_method": row.get("extraction_method", ""),
            }
        )

    return sha256_text(json.dumps(stable_rows, sort_keys=True, ensure_ascii=False))


def group_rows_by_file(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {}

    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)

    return grouped


class SupplierMemoryManager:
    """
    Automatic supplier memory store.

    This stores facts extracted from indexed PDF chunks so users do not have to
    manually maintain the facts layer.
    """

    def __init__(self, db_path: Path = MEMORY_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supplier_memory (
                    file_name TEXT PRIMARY KEY,
                    normalized_file_name TEXT NOT NULL,
                    doc_fingerprint TEXT NOT NULL,
                    supplier_name TEXT,
                    facts_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_supplier_memory_normalized_file
                ON supplier_memory (normalized_file_name)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_supplier_memory_supplier
                ON supplier_memory (supplier_name)
                """
            )

    def get_existing_fingerprint(self, file_name: str) -> str | None:
        normalized = normalize_file_name(file_name)

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT doc_fingerprint
                FROM supplier_memory
                WHERE normalized_file_name = ?
                """,
                (normalized,),
            ).fetchone()

        if not row:
            return None

        return row[0]

    def upsert_facts(self, file_name: str, doc_fingerprint: str, facts: dict[str, Any]):
        now = utc_now()
        supplier_name = None

        try:
            supplier_name = facts.get("supplier_name", {}).get("value")
        except Exception:
            supplier_name = None

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO supplier_memory (
                    file_name,
                    normalized_file_name,
                    doc_fingerprint,
                    supplier_name,
                    facts_json,
                    generated_at,
                    updated_at,
                    hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(file_name) DO UPDATE SET
                    normalized_file_name = excluded.normalized_file_name,
                    doc_fingerprint = excluded.doc_fingerprint,
                    supplier_name = excluded.supplier_name,
                    facts_json = excluded.facts_json,
                    updated_at = excluded.updated_at
                """,
                (
                    file_name,
                    normalize_file_name(file_name),
                    doc_fingerprint,
                    supplier_name,
                    json.dumps(facts, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def refresh(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build or update memory for indexed files.

        If the document fingerprint is unchanged, memory is reused.
        If the document is new or changed, facts are rebuilt automatically.
        """
        grouped = group_rows_by_file(rows)

        refreshed = []
        reused = []
        failed = []

        for file_name, file_rows in sorted(grouped.items()):
            fingerprint = compute_file_fingerprint(file_rows)
            existing_fingerprint = self.get_existing_fingerprint(file_name)

            if existing_fingerprint == fingerprint:
                reused.append(file_name)
                continue

            try:
                facts = extract_facts_for_file(file_name, file_rows)
                self.upsert_facts(
                    file_name=file_name,
                    doc_fingerprint=fingerprint,
                    facts=facts,
                )
                refreshed.append(file_name)
            except Exception as exc:
                failed.append(
                    {
                        "file_name": file_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return {
            "refreshed": refreshed,
            "reused": reused,
            "failed": failed,
            "num_refreshed": len(refreshed),
            "num_reused": len(reused),
            "num_failed": len(failed),
        }

    def get_facts(self, file_name: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """
        Get facts for a file.

        If facts do not exist and rows are provided, this automatically builds
        memory for that file.
        """
        normalized = normalize_file_name(file_name)

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT file_name, facts_json
                FROM supplier_memory
                WHERE normalized_file_name = ?
                """,
                (normalized,),
            ).fetchone()

            if row:
                conn.execute(
                    """
                    UPDATE supplier_memory
                    SET hit_count = hit_count + 1,
                        updated_at = ?
                    WHERE normalized_file_name = ?
                    """,
                    (utc_now(), normalized),
                )

                return json.loads(row[1])

        if rows:
            selected = [
                row for row in rows
                if normalize_file_name(row["file_name"]) == normalized
            ]

            if selected:
                fingerprint = compute_file_fingerprint(selected)
                facts = extract_facts_for_file(file_name, selected)
                self.upsert_facts(file_name, fingerprint, facts)
                return facts

        return None

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM supplier_memory").fetchone()[0]
            total_hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM supplier_memory").fetchone()[0]

            rows = conn.execute(
                """
                SELECT file_name, supplier_name, doc_fingerprint, hit_count, generated_at, updated_at
                FROM supplier_memory
                ORDER BY updated_at DESC
                LIMIT 50
                """
            ).fetchall()

        recent = []

        for row in rows:
            recent.append(
                {
                    "file_name": row[0],
                    "supplier_name": row[1],
                    "doc_fingerprint_short": row[2][:16],
                    "hit_count": int(row[3]),
                    "generated_at": row[4],
                    "updated_at": row[5],
                }
            )

        return {
            "memory_db_path": str(self.db_path),
            "total_memory_records": int(total),
            "total_memory_hits": int(total_hits),
            "recent": recent,
        }
