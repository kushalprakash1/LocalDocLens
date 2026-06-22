import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from localdoc.verifier import sha256_text


LEARNING_DB_PATH = Path("data/db/learning_examples.sqlite")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def compact(text: str) -> str:
    return " ".join(str(text or "").split())


def first_source(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {}

    return evidence[0]


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        compact(item.get("text", ""))
        for item in evidence
        if item.get("text")
    )


class LearningExampleStore:
    """
    Local-only learning example store.

    This does not upload anything.
    It stores answer/evidence/verification records locally so the project can
    later export evaluation or fine-tuning data.
    """

    def __init__(self, db_path: Path = LEARNING_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_examples (
                    example_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    answer_mode TEXT NOT NULL,
                    file_name TEXT,
                    page_number INTEGER,
                    evidence_text TEXT,
                    evidence_json TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    verification_confidence REAL NOT NULL,
                    verification_reason TEXT NOT NULL,
                    question_type TEXT,
                    verification_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learning_status
                ON learning_examples (verification_status)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learning_mode
                ON learning_examples (answer_mode)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learning_file
                ON learning_examples (file_name)
                """
            )

    def make_example_id(
        self,
        question: str,
        answer: str,
        answer_mode: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        source = first_source(evidence)

        raw = json.dumps(
            {
                "question": question.strip().lower(),
                "answer": answer.strip(),
                "answer_mode": answer_mode,
                "file_name": source.get("file_name"),
                "page_number": source.get("page_number"),
                "evidence_hash": sha256_text(evidence_text(evidence)),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        return sha256_text(raw)

    def record_example(
        self,
        question: str,
        answer: str,
        answer_mode: str,
        evidence: list[dict[str, Any]],
        verification: dict[str, Any],
    ) -> str:
        now = utc_now()
        source = first_source(evidence)
        ev_text = evidence_text(evidence)
        example_id = self.make_example_id(
            question=question,
            answer=answer,
            answer_mode=answer_mode,
            evidence=evidence,
        )

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_examples (
                    example_id,
                    question,
                    answer,
                    answer_mode,
                    file_name,
                    page_number,
                    evidence_text,
                    evidence_json,
                    verification_status,
                    verification_confidence,
                    verification_reason,
                    question_type,
                    verification_json,
                    created_at,
                    updated_at,
                    hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(example_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    hit_count = learning_examples.hit_count + 1,
                    verification_status = excluded.verification_status,
                    verification_confidence = excluded.verification_confidence,
                    verification_reason = excluded.verification_reason,
                    verification_json = excluded.verification_json
                """,
                (
                    example_id,
                    question,
                    answer,
                    answer_mode,
                    source.get("file_name"),
                    source.get("page_number"),
                    ev_text,
                    json.dumps(evidence, ensure_ascii=False),
                    verification.get("status", "unverified"),
                    float(verification.get("confidence", 0.0)),
                    verification.get("reason", ""),
                    verification.get("question_type", ""),
                    json.dumps(verification, ensure_ascii=False),
                    now,
                    now,
                ),
            )

        return example_id

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM learning_examples").fetchone()[0]

            by_status_rows = conn.execute(
                """
                SELECT verification_status, COUNT(*)
                FROM learning_examples
                GROUP BY verification_status
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()

            by_mode_rows = conn.execute(
                """
                SELECT answer_mode, COUNT(*)
                FROM learning_examples
                GROUP BY answer_mode
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()

            recent_rows = conn.execute(
                """
                SELECT example_id, question, answer_mode, file_name, page_number,
                       verification_status, verification_confidence, updated_at
                FROM learning_examples
                ORDER BY updated_at DESC
                LIMIT 20
                """
            ).fetchall()

        return {
            "learning_db_path": str(self.db_path),
            "total_examples": int(total),
            "by_status": {row[0]: int(row[1]) for row in by_status_rows},
            "by_answer_mode": {row[0]: int(row[1]) for row in by_mode_rows},
            "recent": [
                {
                    "example_id": row[0],
                    "question": row[1],
                    "answer_mode": row[2],
                    "file_name": row[3],
                    "page_number": row[4],
                    "verification_status": row[5],
                    "verification_confidence": row[6],
                    "updated_at": row[7],
                }
                for row in recent_rows
            ],
        }

    def export_jsonl(
        self,
        output_path: str = "artifacts/training_examples.jsonl",
        include_statuses: list[str] | None = None,
    ) -> dict[str, Any]:
        if include_statuses is None:
            include_statuses = ["auto_verified"]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        placeholders = ",".join("?" for _ in include_statuses)

        query = f"""
            SELECT example_id, question, answer, answer_mode, file_name, page_number,
                   evidence_text, verification_status, verification_confidence,
                   verification_reason, question_type, verification_json
            FROM learning_examples
            WHERE verification_status IN ({placeholders})
            ORDER BY updated_at DESC
        """

        count = 0

        with self.connect() as conn:
            rows = conn.execute(query, include_statuses).fetchall()

        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                verification = json.loads(row[11])

                record = {
                    "example_id": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "answer_mode": row[3],
                    "file_name": row[4],
                    "page_number": row[5],
                    "evidence_text": row[6],
                    "verification_status": row[7],
                    "verification_confidence": row[8],
                    "verification_reason": row[9],
                    "question_type": row[10],
                    "checks": verification.get("checks", {}),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {
            "output_path": str(output),
            "num_examples_exported": count,
            "included_statuses": include_statuses,
        }
