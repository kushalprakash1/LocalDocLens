import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from localdoc.batch_analyze import (
    ANALYSIS_DB_PATH,
    AnalysisMemoryStore,
    build_batch_summary,
    build_document_memory,
    clean_text,
    compute_doc_fingerprint,
    filter_rows,
    group_by_file,
    load_rows,
    make_page_memory,
    render_markdown,
    sha256_text,
)
from localdoc.memory import SupplierMemoryManager


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def compute_page_memory_id(row: dict[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            {
                "file_name": row["file_name"],
                "page_number": row["page_number"],
                "chunk_id": row["chunk_id"],
                "text_hash": sha256_text(row["text"]),
            },
            sort_keys=True,
        )
    )


def compute_page_text_hash(row: dict[str, Any]) -> str:
    return sha256_text(row["text"])


class ResumableAnalysisStore:
    """
    Tracks batch-analysis jobs so large runs can resume safely.
    Uses the same analysis memory DB as page/document memory.
    """

    def __init__(self, db_path: Path = ANALYSIS_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    job_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    use_llm INTEGER NOT NULL,
                    file_filter TEXT,
                    status TEXT NOT NULL,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    processed_files INTEGER NOT NULL DEFAULT 0,
                    processed_pages INTEGER NOT NULL DEFAULT 0,
                    skipped_pages INTEGER NOT NULL DEFAULT 0,
                    processed_documents INTEGER NOT NULL DEFAULT 0,
                    skipped_documents INTEGER NOT NULL DEFAULT 0,
                    failed_items INTEGER NOT NULL DEFAULT 0,
                    error_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_page_status (
                    page_memory_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_document_status (
                    file_name TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    doc_fingerprint TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    use_llm INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_status_file ON analysis_page_status (file_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_status_status ON analysis_document_status (status)")

    def make_job_id(
        self,
        mode: str,
        use_llm: bool,
        file_filter: str | None,
        selected_rows: list[dict[str, Any]],
    ) -> str:
        batch_fingerprint = sha256_text(
            json.dumps(
                [
                    {
                        "file_name": row["file_name"],
                        "page_number": row["page_number"],
                        "chunk_id": row["chunk_id"],
                        "text_hash": sha256_text(row["text"]),
                    }
                    for row in sorted(
                        selected_rows,
                        key=lambda item: (item["file_name"], item["page_number"], item["chunk_id"]),
                    )
                ],
                sort_keys=True,
                ensure_ascii=False,
            )
        )

        return sha256_text(
            json.dumps(
                {
                    "mode": mode,
                    "use_llm": use_llm,
                    "file_filter": file_filter,
                    "batch_fingerprint": batch_fingerprint,
                },
                sort_keys=True,
            )
        )

    def start_job(
        self,
        job_id: str,
        mode: str,
        use_llm: bool,
        file_filter: str | None,
        total_files: int,
        total_pages: int,
        total_chunks: int,
    ):
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_jobs (
                    job_id,
                    mode,
                    use_llm,
                    file_filter,
                    status,
                    total_files,
                    total_pages,
                    total_chunks,
                    processed_files,
                    processed_pages,
                    skipped_pages,
                    processed_documents,
                    skipped_documents,
                    failed_items,
                    error_json,
                    started_at,
                    updated_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, 'running', ?, ?, ?, 0, 0, 0, 0, 0, 0, '[]', ?, ?, NULL)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = 'running',
                    total_files = excluded.total_files,
                    total_pages = excluded.total_pages,
                    total_chunks = excluded.total_chunks,
                    processed_files = 0,
                    processed_pages = 0,
                    skipped_pages = 0,
                    processed_documents = 0,
                    skipped_documents = 0,
                    failed_items = 0,
                    error_json = '[]',
                    updated_at = excluded.updated_at,
                    completed_at = NULL
                """,
                (
                    job_id,
                    mode,
                    1 if use_llm else 0,
                    file_filter,
                    total_files,
                    total_pages,
                    total_chunks,
                    now,
                    now,
                ),
            )

    def complete_job(self, job_id: str):
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE analysis_jobs
                SET status = 'completed',
                    updated_at = ?,
                    completed_at = ?
                WHERE job_id = ?
                """,
                (now, now, job_id),
            )

    def fail_job(self, job_id: str, error: str):
        now = utc_now()

        with self.connect() as conn:
            row = conn.execute(
                "SELECT error_json FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

            errors = []

            if row:
                try:
                    errors = json.loads(row[0])
                except Exception:
                    errors = []

            errors.append({"error": error, "at": now})

            conn.execute(
                """
                UPDATE analysis_jobs
                SET status = 'failed',
                    failed_items = failed_items + 1,
                    error_json = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(errors, ensure_ascii=False), now, job_id),
            )

    def increment(self, job_id: str, field: str, amount: int = 1):
        allowed = {
            "processed_files",
            "processed_pages",
            "skipped_pages",
            "processed_documents",
            "skipped_documents",
            "failed_items",
        }

        if field not in allowed:
            raise RuntimeError(f"Invalid increment field: {field}")

        with self.connect() as conn:
            conn.execute(
                f"""
                UPDATE analysis_jobs
                SET {field} = {field} + ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (amount, utc_now(), job_id),
            )

    def page_memory_exists(self, page_memory_id: str, text_hash: str, row: dict[str, Any] | None = None) -> bool:
        """
        Robust resume check.

        First checks the current page_memory_id.
        Then checks stable row identity in case the page_memory_id format changed:
        file_name + page_number + chunk_id + text_hash.
        """
        with self.connect() as conn:
            direct = conn.execute(
                """
                SELECT page_memory_id
                FROM page_memory
                WHERE page_memory_id = ? AND text_hash = ?
                """,
                (page_memory_id, text_hash),
            ).fetchone()

            if direct is not None:
                return True

            if row is not None:
                stable = conn.execute(
                    """
                    SELECT page_memory_id
                    FROM page_memory
                    WHERE file_name = ?
                      AND page_number = ?
                      AND chunk_id = ?
                      AND text_hash = ?
                    """,
                    (
                        row["file_name"],
                        int(row["page_number"]),
                        row["chunk_id"],
                        text_hash,
                    ),
                ).fetchone()

                if stable is not None:
                    return True

        return False

    def record_page_status(
        self,
        job_id: str,
        row: dict[str, Any],
        page_memory_id: str,
        text_hash: str,
        status: str,
        error: str | None = None,
    ):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_page_status (
                    page_memory_id,
                    job_id,
                    file_name,
                    page_number,
                    chunk_id,
                    text_hash,
                    status,
                    error,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(page_memory_id) DO UPDATE SET
                    job_id = excluded.job_id,
                    text_hash = excluded.text_hash,
                    status = excluded.status,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    page_memory_id,
                    job_id,
                    row["file_name"],
                    row["page_number"],
                    row["chunk_id"],
                    text_hash,
                    status,
                    error,
                    utc_now(),
                ),
            )

    def load_existing_document_memory(self, file_name: str, doc_fingerprint: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT file_name, doc_fingerprint, supplier_name, overall_risk, decision,
                       document_summary, document_types_json, facts_json, findings_json,
                       pages_json, created_at
                FROM document_memory
                WHERE file_name = ? AND doc_fingerprint = ?
                """,
                (file_name, doc_fingerprint),
            ).fetchone()

        if not row:
            return None

        return {
            "file_name": row[0],
            "doc_fingerprint": row[1],
            "supplier_name": row[2],
            "overall_risk": row[3],
            "decision": row[4],
            "document_summary": row[5],
            "document_types": json.loads(row[6]),
            "facts": json.loads(row[7]),
            "findings": json.loads(row[8]),
            "pages": json.loads(row[9]),
            "num_chunks": None,
            "num_pages_with_risk": None,
            "pages_with_risk": [],
            "created_at": row[10],
        }

    def record_document_status(
        self,
        job_id: str,
        file_name: str,
        doc_fingerprint: str,
        mode: str,
        use_llm: bool,
        status: str,
        error: str | None = None,
    ):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_document_status (
                    file_name,
                    job_id,
                    doc_fingerprint,
                    mode,
                    use_llm,
                    status,
                    error,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_name) DO UPDATE SET
                    job_id = excluded.job_id,
                    doc_fingerprint = excluded.doc_fingerprint,
                    mode = excluded.mode,
                    use_llm = excluded.use_llm,
                    status = excluded.status,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    file_name,
                    job_id,
                    doc_fingerprint,
                    mode,
                    1 if use_llm else 0,
                    status,
                    error,
                    utc_now(),
                ),
            )

    def job_stats(self, job_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, mode, use_llm, file_filter, status,
                       total_files, total_pages, total_chunks,
                       processed_files, processed_pages, skipped_pages,
                       processed_documents, skipped_documents, failed_items,
                       started_at, updated_at, completed_at, error_json
                FROM analysis_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

        if not row:
            return {}

        return {
            "job_id": row[0],
            "mode": row[1],
            "use_llm": bool(row[2]),
            "file_filter": row[3],
            "status": row[4],
            "total_files": row[5],
            "total_pages": row[6],
            "total_chunks": row[7],
            "processed_files": row[8],
            "processed_pages": row[9],
            "skipped_pages": row[10],
            "processed_documents": row[11],
            "skipped_documents": row[12],
            "failed_items": row[13],
            "started_at": row[14],
            "updated_at": row[15],
            "completed_at": row[16],
            "errors": json.loads(row[17] or "[]"),
        }

    def latest_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, mode, use_llm, file_filter, status,
                       total_files, total_pages, processed_pages,
                       skipped_pages, failed_items, started_at, updated_at, completed_at
                FROM analysis_jobs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "job_id": row[0],
                "mode": row[1],
                "use_llm": bool(row[2]),
                "file_filter": row[3],
                "status": row[4],
                "total_files": row[5],
                "total_pages": row[6],
                "processed_pages": row[7],
                "skipped_pages": row[8],
                "failed_items": row[9],
                "started_at": row[10],
                "updated_at": row[11],
                "completed_at": row[12],
            }
            for row in rows
        ]


def run_resumable_batch_analysis(
    file_name: str | None = None,
    mode: str = "hybrid",
    use_llm: bool = False,
    output_dir: str = "artifacts",
    force: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()

    mode = mode.lower().strip()

    if mode not in {"fast", "hybrid", "llm"}:
        raise RuntimeError("mode must be one of: fast, hybrid, llm")

    if mode == "llm":
        use_llm = True

    rows = load_rows()
    selected_rows = filter_rows(rows, file_name)
    grouped = group_by_file(selected_rows)

    unique_pages = len({(row["file_name"], row["page_number"]) for row in selected_rows})

    store = AnalysisMemoryStore()
    job_store = ResumableAnalysisStore()
    supplier_memory = SupplierMemoryManager()

    job_id = job_store.make_job_id(
        mode=mode,
        use_llm=use_llm,
        file_filter=file_name,
        selected_rows=selected_rows,
    )

    job_store.start_job(
        job_id=job_id,
        mode=mode,
        use_llm=use_llm,
        file_filter=file_name,
        total_files=len(grouped),
        total_pages=unique_pages,
        total_chunks=len(selected_rows),
    )

    document_memories = []
    all_page_memories = []

    try:
        for grouped_file_name, file_rows in sorted(grouped.items()):
            file_rows = sorted(file_rows, key=lambda row: (row["page_number"], row["chunk_id"]))
            page_memories = []

            for row in file_rows:
                page_memory_id = compute_page_memory_id(row)
                text_hash = compute_page_text_hash(row)

                if resume and not force and job_store.page_memory_exists(page_memory_id, text_hash, row):
                    job_store.record_page_status(
                        job_id=job_id,
                        row=row,
                        page_memory_id=page_memory_id,
                        text_hash=text_hash,
                        status="skipped_reused",
                    )

                    job_store.increment(job_id, "skipped_pages")
                    continue

                try:
                    page_memory = make_page_memory(row)
                    store.upsert_page_memory(page_memory)
                    page_memories.append(page_memory)
                    all_page_memories.append(page_memory)

                    job_store.record_page_status(
                        job_id=job_id,
                        row=row,
                        page_memory_id=page_memory_id,
                        text_hash=text_hash,
                        status="processed",
                    )

                    job_store.increment(job_id, "processed_pages")

                except Exception as exc:
                    job_store.record_page_status(
                        job_id=job_id,
                        row=row,
                        page_memory_id=page_memory_id,
                        text_hash=text_hash,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    job_store.increment(job_id, "failed_items")
                    raise

            # If every page was skipped, rebuild lightweight page memories in memory
            # from current rows so document summaries/findings can still be generated.
            if not page_memories:
                for row in file_rows:
                    page_memories.append(make_page_memory(row))

            doc_fingerprint = compute_doc_fingerprint(file_rows)

            existing_doc = None

            if resume and not force and not use_llm:
                existing_doc = job_store.load_existing_document_memory(grouped_file_name, doc_fingerprint)

            if existing_doc is not None:
                document_memory = existing_doc
                job_store.record_document_status(
                    job_id=job_id,
                    file_name=grouped_file_name,
                    doc_fingerprint=doc_fingerprint,
                    mode=mode,
                    use_llm=use_llm,
                    status="skipped_reused",
                )
                job_store.increment(job_id, "skipped_documents")
            else:
                try:
                    document_memory = build_document_memory(
                        file_name=grouped_file_name,
                        rows=file_rows,
                        page_memories=page_memories,
                        mode=mode,
                        use_llm=use_llm,
                    )

                    store.upsert_document_memory(document_memory)

                    for finding in document_memory["findings"]:
                        store.upsert_finding(finding)

                    supplier_memory.refresh(file_rows)

                    job_store.record_document_status(
                        job_id=job_id,
                        file_name=grouped_file_name,
                        doc_fingerprint=doc_fingerprint,
                        mode=mode,
                        use_llm=use_llm,
                        status="processed",
                    )
                    job_store.increment(job_id, "processed_documents")

                except Exception as exc:
                    job_store.record_document_status(
                        job_id=job_id,
                        file_name=grouped_file_name,
                        doc_fingerprint=doc_fingerprint,
                        mode=mode,
                        use_llm=use_llm,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    job_store.increment(job_id, "failed_items")
                    raise

            document_memories.append(document_memory)
            job_store.increment(job_id, "processed_files")

        summary = build_batch_summary(document_memories)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        duration_s = round(time.perf_counter() - start, 4)

        report = {
            "generated_at": utc_now(),
            "job_id": job_id,
            "mode": mode,
            "use_llm": use_llm,
            "file_filter": file_name,
            "force": force,
            "resume": resume,
            "num_files": len(document_memories),
            "num_pages": unique_pages,
            "num_chunks": len(selected_rows),
            "duration_s": duration_s,
            "analysis_db_path": str(ANALYSIS_DB_PATH),
            "summary": summary,
            "documents": document_memories,
            "page_memory_count_this_run": len(all_page_memories),
            "store_stats": store.stats(),
            "job_stats": job_store.job_stats(job_id),
            "latest_jobs": job_store.latest_jobs(),
        }

        json_path = output_path / "batch_analysis_report.json"
        md_path = output_path / "batch_analysis_report.md"

        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")

        job_store.complete_job(job_id)

        report["output_files"] = {
            "json": str(json_path),
            "markdown": str(md_path),
        }

        report["job_stats"] = job_store.job_stats(job_id)

        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        return report

    except Exception as exc:
        job_store.fail_job(job_id, f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    print(json.dumps(run_resumable_batch_analysis(), indent=2, ensure_ascii=False))
