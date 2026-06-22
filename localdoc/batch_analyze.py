import hashlib
import json
import math
import re
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb

from localdoc.config import LANCEDB_PATH
from localdoc.facts import extract_facts_for_file
from localdoc.memory import SupplierMemoryManager


ANALYSIS_DB_PATH = Path("data/db/analysis_memory.sqlite")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_float(value):
    try:
        result = float(value)

        if math.isnan(result):
            return None

        return result
    except Exception:
        return None


def clean_text(text: str, max_chars: int = 700) -> str:
    text = " ".join(str(text or "").split())

    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."

    return text


def normalize_file_name(file_name: str) -> str:
    return Path(file_name).name.lower().strip()


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


def filter_rows(rows: list[dict[str, Any]], file_name: str | None) -> list[dict[str, Any]]:
    if not file_name:
        return rows

    from localdoc.runtime_security import validate_safe_file_name

    safe_file_name = validate_safe_file_name(file_name)
    target = normalize_file_name(safe_file_name)

    selected = [
        row for row in rows
        if normalize_file_name(row["file_name"]) == target
    ]

    if not selected:
        files = sorted({row["file_name"] for row in rows})
        file_list = "\n".join(f"- {name}" for name in files)

        raise RuntimeError(
            f"No indexed rows found for file: {file_name}\n\n"
            f"Indexed files:\n{file_list}"
        )

    return selected


def group_by_file(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {}

    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)

    return grouped


def compute_doc_fingerprint(rows: list[dict[str, Any]]) -> str:
    stable = []

    for row in sorted(rows, key=lambda item: (item["file_name"], item["page_number"], item["chunk_id"])):
        stable.append(
            {
                "chunk_id": row["chunk_id"],
                "file_name": row["file_name"],
                "page_number": row["page_number"],
                "text": row["text"],
                "extraction_method": row.get("extraction_method", ""),
            }
        )

    return sha256_text(json.dumps(stable, sort_keys=True, ensure_ascii=False))


def guess_document_type(text: str) -> str:
    t = text.lower()

    if "bank verification" in t or "ach" in t or "routing number" in t:
        return "bank_verification"

    if "certificate of insurance" in t or "policy number" in t or "policy carrier" in t:
        return "certificate_of_insurance"

    if "supplier agreement" in t or "agreement effective date" in t or "supplier representative signature" in t:
        return "supplier_agreement"

    if "w-9" in t or "w9" in t or "taxpayer identification" in t:
        return "w9_tax_form"

    if "food safety" in t or "sanctions screening" in t or "conflict of interest" in t:
        return "supplemental_compliance"

    if "onboarding checklist" in t or "supplier legal name" in t or "vendor legal name" in t:
        return "supplier_profile"

    return "unknown"


def extract_key_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}

    emails = re.findall(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text)
    if emails:
        values["emails"] = sorted(set(emails))

    dates = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)
    if dates:
        values["dates"] = sorted(set(dates))

    policy_match = re.search(r"Policy number\s*[:\-]?\s*([A-Z0-9\-]+)", text, flags=re.IGNORECASE)
    if policy_match:
        values["policy_number"] = policy_match.group(1).strip()

    bank_match = re.search(
        r"Bank name\s*[:\-]?\s*([^\n\r]+?)(?:\s+Routing number|\s+Account number|\s+ACH|\s+Payment terms|$)",
        text,
        flags=re.IGNORECASE,
    )
    if bank_match:
        values["bank_name"] = " ".join(bank_match.group(1).split()).strip(" .,:;-")

    routing_match = re.search(r"Routing number\s*[:\-]?\s*([0-9]{6,12})", text, flags=re.IGNORECASE)
    if routing_match:
        values["routing_number"] = routing_match.group(1).strip()

    payment_match = re.search(r"Payment terms\s*[:\-]?\s*(Net\s*\d+)", text, flags=re.IGNORECASE)
    if payment_match:
        values["payment_terms"] = " ".join(payment_match.group(1).split())

    supplier_match = re.search(
        r"(Supplier legal name|Vendor legal name)\s*[:\-]?\s*([^\n\r]+?)(?:\s+DBA|\s+Primary contact|\s+Email|\s+Phone|\s+Bank name|\s+Policy|$)",
        text,
        flags=re.IGNORECASE,
    )
    if supplier_match:
        values["supplier_name"] = " ".join(supplier_match.group(2).split()).strip(" .,:;-")

    signers = re.findall(r"Signed by\s+([A-Z][A-Za-z .'-]+?)\s+on\s+\d{2}/\d{2}/\d{4}", text)
    if signers:
        values["signers"] = sorted(set(" ".join(item.split()).strip() for item in signers))

    return values


def add_signal(signals: list[dict[str, Any]], signal_type: str, severity: str, message: str, evidence: str):
    signals.append(
        {
            "type": signal_type,
            "severity": severity,
            "message": message,
            "evidence": clean_text(evidence, max_chars=500),
        }
    )


def detect_page_risk_signals(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = row["text"]
    t = text.lower()
    signals: list[dict[str, Any]] = []

    # Negation guards:
    # Do NOT flag pages that explicitly say no issue exists.
    no_sanctions_issue = any(
        phrase in t
        for phrase in [
            "no sanctions exception",
            "no sanctions match",
            "no sanctions issue",
            "sanctions screening: clear",
            "sanctions screening clear",
            "sanctions screening result: clear",
            "sanctions screening result clear",
        ]
    )

    # Stress-test filler pages say:
    # "No W-9 exception, insurance exception, bank exception, sanctions exception..."
    # That means every listed exception is absent.
    if (
        "no w-9 exception" in t
        and "insurance exception" in t
        and "bank exception" in t
        and "sanctions exception" in t
        and "supplier agreement exception" in t
    ):
        no_sanctions_issue = True

    no_conflict_issue = any(
        phrase in t
        for phrase in [
            "conflict of interest disclosure: none disclosed",
            "conflict of interest: none disclosed",
            "no conflict of interest",
            "none disclosed",
        ]
    )

    no_bank_issue = any(
        phrase in t
        for phrase in [
            "no bank exception",
            "bank verification status: verified",
            "bank information appears valid",
        ]
    )

    no_agreement_issue = any(
        phrase in t
        for phrase in [
            "no supplier agreement exception",
            "agreement status: signed",
            "signed by both parties",
            "agreement is complete",
        ]
    )

    no_w9_issue = any(
        phrase in t
        for phrase in [
            "no w-9 exception",
            "signed w-9 received",
            "w-9 tax form received",
            "w-9 received",
        ]
    )

    no_insurance_issue = any(
        phrase in t
        for phrase in [
            "no insurance exception",
            "not expired",
            "certificate of insurance is active",
            "insurance is active",
        ]
    )

    if not no_insurance_issue and (
        "appears expired" in t
        or "is expired" in t
        or "coverage expired" in t
        or "insurance expired" in t
    ):
        add_signal(
            signals,
            "expired_insurance",
            "high",
            "Insurance appears expired.",
            text,
        )

    if not no_w9_issue and (
        "w-9 tax form missing" in t
        or "w-9 missing" in t
        or "missing w-9" in t
        or "must provide signed w-9" in t
    ):
        add_signal(
            signals,
            "missing_w9",
            "high",
            "W-9 appears missing or not received.",
            text,
        )

    if not no_agreement_issue and (
        "signature missing" in t
        or "missing supplier signature" in t
        or "supplier representative signature missing" in t
        or "pending signature" in t
    ):
        add_signal(
            signals,
            "missing_signature",
            "high",
            "Supplier agreement signature appears missing.",
            text,
        )

    if not no_bank_issue and "not verified" in t and ("bank" in t or "ach" in t):
        add_signal(
            signals,
            "bank_not_verified",
            "medium",
            "Bank verification may be incomplete.",
            text,
        )

    if not no_conflict_issue and "conflict of interest" in t and (
        "yes" in t
        or "disclosed" in t
        or "possible conflict" in t
        or "conflict identified" in t
    ):
        add_signal(
            signals,
            "conflict_of_interest",
            "medium",
            "Conflict of interest disclosure may need review.",
            text,
        )

    # Important:
    # Do NOT use generic "review" as a sanctions trigger.
    # Many safe pages contain "procurement review note."
    strong_sanctions_signal = any(
        phrase in t
        for phrase in [
            "sanctions screening: possible match",
            "sanctions screening possible match",
            "possible sanctions match",
            "sanctions possible match",
            "sanctions screening requires review",
            "sanctions screening needs review",
            "sanctions screening flagged",
            "sanctions flag",
            "sanctions match",
        ]
    )

    if not no_sanctions_issue and strong_sanctions_signal:
        add_signal(
            signals,
            "sanctions_review",
            "high",
            "Sanctions screening may need review.",
            text,
        )

    ocr_confidence = row.get("ocr_confidence")

    if ocr_confidence is not None:
        try:
            if float(ocr_confidence) < 0.80:
                add_signal(
                    signals,
                    "low_ocr_confidence",
                    "medium",
                    f"OCR confidence is low: {ocr_confidence}",
                    text,
                )
        except Exception:
            pass

    return signals

def summarize_page(row: dict[str, Any], document_type: str, key_values: dict[str, Any], risk_signals: list[dict[str, Any]]) -> str:
    parts = []

    if document_type != "unknown":
        parts.append(f"Page appears to contain {document_type.replace('_', ' ')} information.")
    else:
        parts.append("Page document type is unclear.")

    if key_values:
        keys = ", ".join(sorted(key_values.keys()))
        parts.append(f"Detected key fields: {keys}.")

    if risk_signals:
        risks = ", ".join(sorted({signal["type"] for signal in risk_signals}))
        parts.append(f"Risk signals detected: {risks}.")
    else:
        parts.append("No obvious page-level risk signal detected.")

    return " ".join(parts)


def make_page_memory(row: dict[str, Any]) -> dict[str, Any]:
    document_type = guess_document_type(row["text"])
    key_values = extract_key_values(row["text"])
    risk_signals = detect_page_risk_signals(row)
    page_summary = summarize_page(row, document_type, key_values, risk_signals)

    page_memory_id = sha256_text(
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

    return {
        "page_memory_id": page_memory_id,
        "file_name": row["file_name"],
        "page_number": row["page_number"],
        "chunk_id": row["chunk_id"],
        "text_hash": sha256_text(row["text"]),
        "extraction_method": row["extraction_method"],
        "ocr_confidence": row["ocr_confidence"],
        "document_type": document_type,
        "key_values": key_values,
        "risk_signals": risk_signals,
        "page_summary": page_summary,
        "evidence_text": clean_text(row["text"], max_chars=1500),
        "created_at": utc_now(),
    }


def facts_value(facts: dict[str, Any], path: list[str]) -> Any:
    current: Any = facts

    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None

        current = current[part]

    if isinstance(current, dict) and "value" in current:
        return current.get("value")

    return current


def make_finding(
    file_name: str,
    finding_type: str,
    severity: str,
    status: str,
    message: str,
    page_number: int | None,
    evidence_text: str | None,
) -> dict[str, Any]:
    raw = json.dumps(
        {
            "file_name": file_name,
            "finding_type": finding_type,
            "severity": severity,
            "status": status,
            "message": message,
            "page_number": page_number,
            "evidence_hash": sha256_text(evidence_text or ""),
        },
        sort_keys=True,
    )

    return {
        "finding_id": sha256_text(raw),
        "file_name": file_name,
        "finding_type": finding_type,
        "severity": severity,
        "status": status,
        "message": message,
        "page_number": page_number,
        "evidence_text": clean_text(evidence_text or "", max_chars=1000),
        "verified_by_evidence": bool(page_number and evidence_text),
    }


def findings_from_facts(file_name: str, facts: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    risk = facts.get("risk", {})
    failed_items = risk.get("failed_items", []) or []
    review_items = risk.get("review_items", []) or []

    fact_sources = []

    def collect_sources(obj: Any):
        if isinstance(obj, dict):
            if obj.get("source") and obj.get("evidence_text"):
                fact_sources.append(obj)

            for value in obj.values():
                collect_sources(value)

        elif isinstance(obj, list):
            for value in obj:
                collect_sources(value)

    collect_sources(facts)

    def classify_item(item: str) -> tuple[str, list[str]]:
        item_lower = item.lower()

        if "insurance" in item_lower:
            return (
                "expired_insurance",
                [
                    "certificate of insurance",
                    "policy expiration date",
                    "appears expired",
                    "insurance appears expired",
                    "coverage expired",
                ],
            )

        if "w-9" in item_lower or "w9" in item_lower:
            return (
                "missing_w9",
                [
                    "w-9 tax form missing",
                    "w-9 missing",
                    "missing w-9",
                    "must provide signed w-9",
                    "taxpayer identification section",
                ],
            )

        if "signature" in item_lower or "agreement" in item_lower:
            return (
                "missing_signature",
                [
                    "supplier representative signature missing",
                    "missing supplier signature",
                    "signature missing",
                    "pending signature",
                    "supplier agreement signature page",
                ],
            )

        if "bank" in item_lower:
            return (
                "bank_not_verified",
                [
                    "bank verification",
                    "not verified",
                    "ach",
                    "routing number",
                ],
            )

        if "sanctions" in item_lower:
            return (
                "sanctions_review",
                [
                    "sanctions screening: possible match",
                    "possible sanctions match",
                    "sanctions screening requires review",
                    "sanctions flag",
                ],
            )

        safe = item_lower.replace(" ", "_").replace("-", "_")
        safe = re.sub(r"[^a-z0-9_]+", "", safe).strip("_") or "review_item"

        return safe, [item_lower]

    def find_source_for_keywords(keywords: list[str]):
        best_fact = None
        best_score = -1

        for fact in fact_sources:
            evidence = str(fact.get("evidence_text", ""))
            evidence_lower = evidence.lower()

            score = 0

            for keyword in keywords:
                keyword_lower = keyword.lower()

                if keyword_lower in evidence_lower:
                    score += 3

                for token in keyword_lower.split():
                    if len(token) >= 4 and token in evidence_lower:
                        score += 1

            if score > best_score:
                best_score = score
                best_fact = fact

        if best_fact and best_score > 0:
            return best_fact.get("source", {}), str(best_fact.get("evidence_text", ""))

        return {}, ""

    for item in failed_items:
        finding_type, keywords = classify_item(str(item))
        source, evidence = find_source_for_keywords(keywords)

        findings.append(
            make_finding(
                file_name=file_name,
                finding_type=finding_type,
                severity="high",
                status="failed",
                message=f"Failed compliance item: {item}",
                page_number=source.get("page_number"),
                evidence_text=evidence,
            )
        )

    for item in review_items:
        finding_type, keywords = classify_item(str(item))
        source, evidence = find_source_for_keywords(keywords)

        findings.append(
            make_finding(
                file_name=file_name,
                finding_type=finding_type,
                severity="medium",
                status="needs_review",
                message=f"Review item: {item}",
                page_number=source.get("page_number"),
                evidence_text=evidence,
            )
        )

    return findings

def findings_from_page_memory(page_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []

    for page in page_memories:
        for signal in page["risk_signals"]:
            findings.append(
                make_finding(
                    file_name=page["file_name"],
                    finding_type=signal["type"],
                    severity=signal["severity"],
                    status="detected",
                    message=signal["message"],
                    page_number=page["page_number"],
                    evidence_text=signal["evidence"],
                )
            )

    return findings


def call_ollama(prompt: str, timeout: int = 120) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.8,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    answer = result.get("response", "").strip()
    answer = re.sub(r"<think>[\s\S]*?</think>", "", answer, flags=re.IGNORECASE).strip()

    return answer


def build_llm_document_summary(file_name: str, facts: dict[str, Any], page_memories: list[dict[str, Any]]) -> str | None:
    page_summaries = "\n".join(
        f"- Page {page['page_number']}: {page['page_summary']}"
        for page in page_memories[:30]
    )

    facts_brief = {
        "supplier_name": facts_value(facts, ["supplier_name"]),
        "overall_risk": facts.get("risk", {}).get("overall_risk"),
        "decision": facts.get("risk", {}).get("decision"),
        "failed_items": facts.get("risk", {}).get("failed_items"),
        "review_items": facts.get("risk", {}).get("review_items"),
        "insurance_status": facts_value(facts, ["insurance", "status"]),
        "w9_status": facts_value(facts, ["w9_status"]),
        "agreement_status": facts_value(facts, ["agreement", "status"]),
        "bank_status": facts_value(facts, ["bank", "verification_status"]),
    }

    prompt = f"""
You are LocalDocLens, a local supplier document analysis system.

Create a concise supplier document summary using only the facts and page summaries below.
Do not invent anything. Mention source pages when helpful.

File:
{file_name}

Extracted facts:
{json.dumps(facts_brief, indent=2)}

Page summaries:
{page_summaries}

Return:
1. Supplier overview
2. Key compliance status
3. Main risks
4. Recommended next action
""".strip()

    try:
        return call_ollama(prompt)
    except Exception:
        return None


def build_document_memory(
    file_name: str,
    rows: list[dict[str, Any]],
    page_memories: list[dict[str, Any]],
    mode: str,
    use_llm: bool,
) -> dict[str, Any]:
    facts = extract_facts_for_file(file_name, rows)

    supplier_name = facts_value(facts, ["supplier_name"])
    overall_risk = facts.get("risk", {}).get("overall_risk", "Unknown")
    decision = facts.get("risk", {}).get("decision", "")

    findings = []
    findings.extend(findings_from_facts(file_name, facts))
    findings.extend(findings_from_page_memory(page_memories))

    unique_findings = {}

    def finding_priority(finding: dict[str, Any]) -> int:
        # Prefer fact-level failed findings over page-level duplicate signals.
        status = finding.get("status", "")
        if status == "failed":
            return 0
        if status == "needs_review":
            return 1
        if status == "detected":
            return 2
        return 3

    for finding in findings:
        key = (
            finding.get("file_name"),
            finding.get("finding_type"),
            finding.get("page_number"),
        )

        if key not in unique_findings:
            unique_findings[key] = finding
            continue

        if finding_priority(finding) < finding_priority(unique_findings[key]):
            unique_findings[key] = finding

    findings = list(unique_findings.values())

    document_types = sorted({page["document_type"] for page in page_memories})
    pages_with_risk = sorted({page["page_number"] for page in page_memories if page["risk_signals"]})

    deterministic_summary = (
        f"{supplier_name or file_name} has overall risk {overall_risk}. "
        f"{decision} "
        f"Detected document types: {', '.join(document_types)}. "
        f"Pages with risk signals: {pages_with_risk if pages_with_risk else 'none'}."
    )

    llm_summary = None

    if use_llm or mode == "llm":
        llm_summary = build_llm_document_summary(file_name, facts, page_memories)

    document_summary = llm_summary or deterministic_summary

    doc_fingerprint = compute_doc_fingerprint(rows)

    return {
        "file_name": file_name,
        "doc_fingerprint": doc_fingerprint,
        "supplier_name": supplier_name,
        "overall_risk": overall_risk,
        "decision": decision,
        "document_summary": document_summary,
        "document_types": document_types,
        "pages": sorted({row["page_number"] for row in rows}),
        "num_chunks": len(rows),
        "num_pages_with_risk": len(pages_with_risk),
        "pages_with_risk": pages_with_risk,
        "facts": facts,
        "findings": findings,
        "created_at": utc_now(),
    }


class AnalysisMemoryStore:
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
                CREATE TABLE IF NOT EXISTS page_memory (
                    page_memory_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    extraction_method TEXT,
                    ocr_confidence REAL,
                    document_type TEXT,
                    key_values_json TEXT NOT NULL,
                    risk_signals_json TEXT NOT NULL,
                    page_summary TEXT NOT NULL,
                    evidence_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_memory (
                    file_name TEXT PRIMARY KEY,
                    doc_fingerprint TEXT NOT NULL,
                    supplier_name TEXT,
                    overall_risk TEXT,
                    decision TEXT,
                    document_summary TEXT,
                    document_types_json TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    pages_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_findings (
                    finding_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    finding_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    page_number INTEGER,
                    evidence_text TEXT,
                    verified_by_evidence INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    file_filter TEXT,
                    num_files INTEGER NOT NULL,
                    num_pages INTEGER NOT NULL,
                    num_chunks INTEGER NOT NULL,
                    num_findings INTEGER NOT NULL,
                    duration_s REAL NOT NULL,
                    report_json_path TEXT,
                    report_md_path TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_memory_file ON page_memory (file_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_memory_type ON page_memory (document_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_file ON batch_findings (file_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_severity ON batch_findings (severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_memory_risk ON document_memory (overall_risk)")

    def upsert_page_memory(self, page: dict[str, Any]):
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO page_memory (
                    page_memory_id,
                    file_name,
                    page_number,
                    chunk_id,
                    text_hash,
                    extraction_method,
                    ocr_confidence,
                    document_type,
                    key_values_json,
                    risk_signals_json,
                    page_summary,
                    evidence_text,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(page_memory_id) DO UPDATE SET
                    extraction_method = excluded.extraction_method,
                    ocr_confidence = excluded.ocr_confidence,
                    document_type = excluded.document_type,
                    key_values_json = excluded.key_values_json,
                    risk_signals_json = excluded.risk_signals_json,
                    page_summary = excluded.page_summary,
                    evidence_text = excluded.evidence_text,
                    updated_at = excluded.updated_at
                """,
                (
                    page["page_memory_id"],
                    page["file_name"],
                    page["page_number"],
                    page["chunk_id"],
                    page["text_hash"],
                    page["extraction_method"],
                    page["ocr_confidence"],
                    page["document_type"],
                    json.dumps(page["key_values"], ensure_ascii=False),
                    json.dumps(page["risk_signals"], ensure_ascii=False),
                    page["page_summary"],
                    page["evidence_text"],
                    page["created_at"],
                    now,
                ),
            )

    def upsert_document_memory(self, doc: dict[str, Any]):
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO document_memory (
                    file_name,
                    doc_fingerprint,
                    supplier_name,
                    overall_risk,
                    decision,
                    document_summary,
                    document_types_json,
                    facts_json,
                    findings_json,
                    pages_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_name) DO UPDATE SET
                    doc_fingerprint = excluded.doc_fingerprint,
                    supplier_name = excluded.supplier_name,
                    overall_risk = excluded.overall_risk,
                    decision = excluded.decision,
                    document_summary = excluded.document_summary,
                    document_types_json = excluded.document_types_json,
                    facts_json = excluded.facts_json,
                    findings_json = excluded.findings_json,
                    pages_json = excluded.pages_json,
                    updated_at = excluded.updated_at
                """,
                (
                    doc["file_name"],
                    doc["doc_fingerprint"],
                    doc["supplier_name"],
                    doc["overall_risk"],
                    doc["decision"],
                    doc["document_summary"],
                    json.dumps(doc["document_types"], ensure_ascii=False),
                    json.dumps(doc["facts"], ensure_ascii=False),
                    json.dumps(doc["findings"], ensure_ascii=False),
                    json.dumps(doc["pages"], ensure_ascii=False),
                    doc["created_at"],
                    now,
                ),
            )

    def upsert_finding(self, finding: dict[str, Any]):
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_findings (
                    finding_id,
                    file_name,
                    finding_type,
                    severity,
                    status,
                    message,
                    page_number,
                    evidence_text,
                    verified_by_evidence,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(finding_id) DO UPDATE SET
                    severity = excluded.severity,
                    status = excluded.status,
                    message = excluded.message,
                    page_number = excluded.page_number,
                    evidence_text = excluded.evidence_text,
                    verified_by_evidence = excluded.verified_by_evidence,
                    updated_at = excluded.updated_at
                """,
                (
                    finding["finding_id"],
                    finding["file_name"],
                    finding["finding_type"],
                    finding["severity"],
                    finding["status"],
                    finding["message"],
                    finding["page_number"],
                    finding["evidence_text"],
                    1 if finding["verified_by_evidence"] else 0,
                    utc_now(),
                    now,
                ),
            )

    def record_run(self, run: dict[str, Any]):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_runs (
                    run_id,
                    mode,
                    file_filter,
                    num_files,
                    num_pages,
                    num_chunks,
                    num_findings,
                    duration_s,
                    report_json_path,
                    report_md_path,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["run_id"],
                    run["mode"],
                    run["file_filter"],
                    run["num_files"],
                    run["num_pages"],
                    run["num_chunks"],
                    run["num_findings"],
                    run["duration_s"],
                    run["report_json_path"],
                    run["report_md_path"],
                    run["created_at"],
                ),
            )

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            page_count = conn.execute("SELECT COUNT(*) FROM page_memory").fetchone()[0]
            doc_count = conn.execute("SELECT COUNT(*) FROM document_memory").fetchone()[0]
            finding_count = conn.execute("SELECT COUNT(*) FROM batch_findings").fetchone()[0]

            findings_by_severity = conn.execute(
                """
                SELECT severity, COUNT(*)
                FROM batch_findings
                GROUP BY severity
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()

        return {
            "analysis_db_path": str(self.db_path),
            "page_memory_records": int(page_count),
            "document_memory_records": int(doc_count),
            "finding_records": int(finding_count),
            "findings_by_severity": {row[0]: int(row[1]) for row in findings_by_severity},
        }


def build_batch_summary(document_memories: list[dict[str, Any]]) -> dict[str, Any]:
    by_risk: dict[str, int] = {}
    all_findings = []

    for doc in document_memories:
        risk = doc.get("overall_risk") or "Unknown"
        by_risk[risk] = by_risk.get(risk, 0) + 1
        all_findings.extend(doc.get("findings", []))

    high_findings = [finding for finding in all_findings if finding["severity"] == "high"]
    medium_findings = [finding for finding in all_findings if finding["severity"] == "medium"]

    suppliers = []

    for doc in document_memories:
        suppliers.append(
            {
                "file_name": doc["file_name"],
                "supplier_name": doc["supplier_name"],
                "overall_risk": doc["overall_risk"],
                "decision": doc["decision"],
                "num_findings": len(doc["findings"]),
                "num_high_findings": len([f for f in doc["findings"] if f["severity"] == "high"]),
                "pages_with_risk": doc["pages_with_risk"],
            }
        )

    suppliers = sorted(
        suppliers,
        key=lambda item: (
            {"High": 0, "Medium": 1, "Low": 2}.get(item["overall_risk"], 3),
            -item["num_high_findings"],
            item["file_name"],
        ),
    )

    return {
        "num_documents": len(document_memories),
        "risk_distribution": by_risk,
        "num_findings": len(all_findings),
        "num_high_findings": len(high_findings),
        "num_medium_findings": len(medium_findings),
        "suppliers": suppliers,
        "top_findings": sorted(
            all_findings,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}.get(item["severity"], 3),
                item["file_name"],
            ),
        )[:25],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = []

    lines.append("# LocalDocLens Batch Analysis Report")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append(f"Mode: {report['mode']}")
    lines.append(f"Files analyzed: {report['summary']['num_documents']}")
    lines.append(f"Total chunks: {report['num_chunks']}")
    lines.append(f"Total pages: {report['num_pages']}")
    lines.append(f"Duration: {report['duration_s']} seconds")
    lines.append("")

    lines.append("## Risk Distribution")
    lines.append("")

    for risk, count in sorted(report["summary"]["risk_distribution"].items()):
        lines.append(f"- {risk}: {count}")

    lines.append("")
    lines.append("## Suppliers")
    lines.append("")

    for supplier in report["summary"]["suppliers"]:
        lines.append(f"### {supplier['supplier_name'] or supplier['file_name']}")
        lines.append(f"- File: {supplier['file_name']}")
        lines.append(f"- Overall risk: {supplier['overall_risk']}")
        lines.append(f"- Decision: {supplier['decision']}")
        lines.append(f"- Findings: {supplier['num_findings']}")
        lines.append(f"- High findings: {supplier['num_high_findings']}")
        lines.append(f"- Pages with risk: {supplier['pages_with_risk'] if supplier['pages_with_risk'] else 'none'}")
        lines.append("")

    lines.append("## Top Evidence-Backed Findings")
    lines.append("")

    for finding in report["summary"]["top_findings"]:
        lines.append(f"- [{finding['severity'].upper()}] {finding['file_name']} page {finding['page_number']}: {finding['message']}")
        lines.append(f"  - Type: {finding['finding_type']}")
        lines.append(f"  - Evidence: {finding['evidence_text']}")
        lines.append("")

    return "\n".join(lines)


def run_batch_analysis(
    file_name: str | None = None,
    mode: str = "hybrid",
    use_llm: bool = False,
    output_dir: str = "artifacts",
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

    store = AnalysisMemoryStore()
    supplier_memory = SupplierMemoryManager()

    document_memories = []
    all_page_memories = []

    for grouped_file_name, file_rows in sorted(grouped.items()):
        file_rows = sorted(file_rows, key=lambda row: (row["page_number"], row["chunk_id"]))

        page_memories = []

        for row in file_rows:
            page_memory = make_page_memory(row)
            page_memories.append(page_memory)
            all_page_memories.append(page_memory)
            store.upsert_page_memory(page_memory)

        document_memory = build_document_memory(
            file_name=grouped_file_name,
            rows=file_rows,
            page_memories=page_memories,
            mode=mode,
            use_llm=use_llm,
        )

        document_memories.append(document_memory)
        store.upsert_document_memory(document_memory)

        for finding in document_memory["findings"]:
            store.upsert_finding(finding)

        # Keep automatic supplier memory synced too.
        supplier_memory.refresh(file_rows)

    summary = build_batch_summary(document_memories)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    duration_s = round(time.perf_counter() - start, 4)

    report = {
        "generated_at": utc_now(),
        "mode": mode,
        "use_llm": use_llm,
        "file_filter": file_name,
        "num_files": len(document_memories),
        "num_pages": len({(row["file_name"], row["page_number"]) for row in selected_rows}),
        "num_chunks": len(selected_rows),
        "duration_s": duration_s,
        "analysis_db_path": str(ANALYSIS_DB_PATH),
        "summary": summary,
        "documents": document_memories,
        "page_memory_count": len(all_page_memories),
        "store_stats": store.stats(),
    }

    json_path = output_path / "batch_analysis_report.json"
    md_path = output_path / "batch_analysis_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    run = {
        "run_id": sha256_text(json.dumps({"generated_at": report["generated_at"], "duration_s": duration_s}, sort_keys=True)),
        "mode": mode,
        "file_filter": file_name,
        "num_files": report["num_files"],
        "num_pages": report["num_pages"],
        "num_chunks": report["num_chunks"],
        "num_findings": summary["num_findings"],
        "duration_s": duration_s,
        "report_json_path": str(json_path),
        "report_md_path": str(md_path),
        "created_at": utc_now(),
    }

    store.record_run(run)

    report["output_files"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


if __name__ == "__main__":
    print(json.dumps(run_batch_analysis(), indent=2, ensure_ascii=False))
