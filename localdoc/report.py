import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb

from localdoc.config import LANCEDB_PATH


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def safe_float(value):
    try:
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def clean_text(text: str, max_chars: int = 420) -> str:
    text = " ".join(str(text).split())
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def extract_supplier(text: str) -> str:
    """
    Extract supplier name from document text using generic evidence patterns.

    This intentionally avoids hard-coded supplier names so the system works on
    newly uploaded PDFs instead of only the sample packet.
    """
    patterns = [
        r"Supplier legal name\s*[:\-]?\s*([^\n\r]+)",
        r"Insured supplier\s*[:\-]?\s*([^\n\r]+)",
        r"Supplier name\s*[:\-]?\s*([^\n\r]+)",
        r"Legal name\s*[:\-]?\s*([^\n\r]+)",
        r"Vendor legal name\s*[:\-]?\s*([^\n\r]+)",
        r"Vendor name\s*[:\-]?\s*([^\n\r]+)",
        r"Company name\s*[:\-]?\s*([^\n\r]+)",
    ]

    stop_words = [
        "DBA",
        "Bank name",
        "Policy",
        "Document type",
        "Coverage",
        "Field",
        "Primary contact",
        "Email",
        "Phone",
        "Address",
        "Tax",
        "W-9",
        "Certificate",
        "ACH",
        "Agreement",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        value = match.group(1).strip()

        for stop_word in stop_words:
            value = re.split(r"\s+" + re.escape(stop_word) + r"\b", value, flags=re.IGNORECASE)[0].strip()

        value = value.strip(" .:-")

        if value and len(value) >= 2:
            return value

    # Fallback: look for common company suffixes near the beginning of the packet.
    first_part = text[:2500]

    company_match = re.search(
        r"([A-Z][A-Za-z0-9 &'.,-]{2,80}\s+(LLC|Inc\.?|Corporation|Corp\.?|Ltd\.?|Limited|Co\.?|Company))",
        first_part,
    )

    if company_match:
        return company_match.group(1).strip(" .:-")

    return "Unknown supplier"


def extract_dates(text: str) -> list[datetime]:
    dates = []

    for value in re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text):
        try:
            dates.append(datetime.strptime(value, "%m/%d/%Y"))
        except Exception:
            pass

    return dates


def latest_date_text(text: str) -> str | None:
    date_values = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)

    if not date_values:
        return None

    parsed = []

    for value in date_values:
        try:
            parsed.append((datetime.strptime(value, "%m/%d/%Y"), value))
        except Exception:
            pass

    if not parsed:
        return date_values[-1]

    parsed.sort(key=lambda x: x[0])
    return parsed[-1][1]


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


def filter_rows_by_file(rows: list[dict[str, Any]], file_name: str | None) -> list[dict[str, Any]]:
    if not file_name:
        files = sorted({row["file_name"] for row in rows})

        if len(files) <= 1:
            return rows

        file_list = "\n".join(f"- {name}" for name in files)

        raise RuntimeError(
            "Multiple files are indexed. To avoid mixing suppliers, run:\n\n"
            "localdoc report --file \"YOUR_FILE_NAME.pdf\"\n\n"
            f"Indexed files:\n{file_list}"
        )

    target = Path(file_name).name.lower()

    selected = [
        row for row in rows
        if Path(row["file_name"]).name.lower() == target or row["file_name"].lower() == file_name.lower()
    ]

    if not selected:
        files = sorted({row["file_name"] for row in rows})
        file_list = "\n".join(f"- {name}" for name in files)

        raise RuntimeError(
            f"No indexed rows found for file: {file_name}\n\n"
            f"Indexed files:\n{file_list}"
        )

    return selected


def score_row(row: dict[str, Any], required: list[str], preferred: list[str]) -> float:
    text_lower = row["text"].lower()

    for term in required:
        if term.lower() not in text_lower:
            return -1.0

    score = 0.0

    for term in preferred:
        if term.lower() in text_lower:
            score += 1.0

    score += min(len(row["text"]) / 5000, 0.25)

    return score


def find_best_row(rows: list[dict[str, Any]], required: list[str], preferred: list[str]) -> dict[str, Any] | None:
    best_row = None
    best_score = -1.0

    for row in rows:
        score = score_row(row, required, preferred)

        if score > best_score:
            best_score = score
            best_row = row

    if best_score < 0:
        return None

    return best_row


def make_finding(
    finding_id: str,
    title: str,
    status: str,
    severity: str,
    row: dict[str, Any] | None,
    recommendation: str,
    details: str,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "title": title,
        "status": status,
        "severity": severity,
        "page_number": row["page_number"] if row else None,
        "file_name": row["file_name"] if row else None,
        "evidence_quote": clean_text(row["text"]) if row else None,
        "recommendation": recommendation,
        "details": details,
    }


def build_supplier_report(output_dir: str = "artifacts", file_name: str | None = None) -> dict[str, Any]:
    all_rows = load_rows()

    if not all_rows:
        raise RuntimeError("No indexed chunks found. Run localdoc ingest data/docs first.")

    rows = filter_rows_by_file(all_rows, file_name)

    joined_text = "\n".join(row["text"] for row in rows)
    supplier = extract_supplier(joined_text)

    source_file = sorted({row["file_name"] for row in rows})[0]
    safe_file_stem = Path(source_file).stem.replace(" ", "_")

    insurance_row = find_best_row(
        rows,
        required=["insurance"],
        preferred=[
            "certificate of insurance",
            "policy expiration date",
            "appears expired",
            "expired",
            "additional insured",
        ],
    )

    w9_row = find_best_row(
        rows,
        required=["w-9"],
        preferred=[
            "missing",
            "not received",
            "signed w-9",
            "before approval",
        ],
    )

    signature_row = find_best_row(
        rows,
        required=["signature"],
        preferred=[
            "supplier agreement",
            "supplier representative signature",
            "missing",
            "pending signature",
        ],
    )

    bank_row = find_best_row(
        rows,
        required=["bank verification"],
        preferred=[
            "verified",
            "ach",
            "bank letter",
            "no bank account change",
        ],
    )

    findings = []

    insurance_status = "unknown"
    insurance_severity = "medium"
    insurance_details = "Insurance status could not be fully determined."

    if insurance_row:
        insurance_text_lower = insurance_row["text"].lower()
        insurance_latest_date = latest_date_text(insurance_row["text"])
        parsed_dates = extract_dates(insurance_row["text"])
        latest_parsed_date = max(parsed_dates) if parsed_dates else None

        is_expired = "expired" in insurance_text_lower

        if latest_parsed_date and latest_parsed_date.date() < datetime.utcnow().date():
            is_expired = True

        if is_expired:
            insurance_status = "failed"
            insurance_severity = "high"
            insurance_details = f"Certificate of Insurance appears expired. Latest detected policy date: {insurance_latest_date}."
        else:
            insurance_status = "passed"
            insurance_severity = "low"
            insurance_details = f"Certificate of Insurance found. Latest detected policy date: {insurance_latest_date}."

    findings.append(
        make_finding(
            finding_id="expired_insurance",
            title="Insurance Coverage",
            status=insurance_status,
            severity=insurance_severity,
            row=insurance_row,
            recommendation="Request an updated Certificate of Insurance before approval."
            if insurance_status == "failed"
            else "No immediate action required for insurance.",
            details=insurance_details,
        )
    )

    w9_status = "unknown"
    w9_severity = "medium"
    w9_details = "W-9 status could not be fully determined."

    if w9_row:
        w9_text_lower = w9_row["text"].lower()

        if "missing" in w9_text_lower or "not received" in w9_text_lower:
            w9_status = "failed"
            w9_severity = "high"
            w9_details = "W-9 tax form is missing or has not been received."
        else:
            w9_status = "passed"
            w9_severity = "low"
            w9_details = "W-9 evidence was found."

    findings.append(
        make_finding(
            finding_id="missing_w9",
            title="W-9 Tax Form",
            status=w9_status,
            severity=w9_severity,
            row=w9_row,
            recommendation="Collect a signed W-9 before supplier approval."
            if w9_status == "failed"
            else "No immediate action required for W-9.",
            details=w9_details,
        )
    )

    signature_status = "unknown"
    signature_severity = "medium"
    signature_details = "Supplier agreement signature status could not be fully determined."

    if signature_row:
        signature_text_lower = signature_row["text"].lower()

        if "missing" in signature_text_lower or "pending signature" in signature_text_lower:
            signature_status = "failed"
            signature_severity = "high"
            signature_details = "Supplier agreement is missing the supplier signature."
        else:
            signature_status = "passed"
            signature_severity = "low"
            signature_details = "Supplier agreement signature evidence was found."

    findings.append(
        make_finding(
            finding_id="missing_signature",
            title="Supplier Agreement Signature",
            status=signature_status,
            severity=signature_severity,
            row=signature_row,
            recommendation="Collect the supplier representative signature before approval."
            if signature_status == "failed"
            else "No immediate action required for supplier agreement signature.",
            details=signature_details,
        )
    )

    bank_status = "unknown"
    bank_severity = "medium"
    bank_details = "Bank verification status could not be fully determined."

    if bank_row:
        bank_text_lower = bank_row["text"].lower()

        if "verified" in bank_text_lower:
            bank_status = "passed"
            bank_severity = "low"
            bank_details = "Bank verification is marked as verified."
        else:
            bank_status = "review"
            bank_severity = "medium"
            bank_details = "Bank verification evidence exists but is not clearly verified."

    findings.append(
        make_finding(
            finding_id="bank_verification",
            title="Bank Verification",
            status=bank_status,
            severity=bank_severity,
            row=bank_row,
            recommendation="No immediate action required for bank verification."
            if bank_status == "passed"
            else "Review bank verification before payment setup.",
            details=bank_details,
        )
    )

    failed_findings = [item for item in findings if item["status"] == "failed"]
    review_findings = [item for item in findings if item["status"] in {"review", "unknown"}]

    if failed_findings:
        overall_risk = "High"
        decision = "Do not approve until failed compliance items are resolved."
    elif review_findings:
        overall_risk = "Medium"
        decision = "Hold approval until unclear compliance items are reviewed."
    else:
        overall_risk = "Low"
        decision = "Supplier appears ready for approval based on available evidence."

    report = {
        "generated_at": utc_now(),
        "supplier": supplier,
        "source_file": source_file,
        "overall_risk": overall_risk,
        "decision": decision,
        "findings": findings,
        "source": {
            "indexed_chunks": len(rows),
            "files": sorted({row["file_name"] for row in rows}),
            "pages": sorted({row["page_number"] for row in rows}),
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{safe_file_stem}_supplier_compliance_report.json"
    md_path = output_path / f"{safe_file_stem}_supplier_compliance_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    legacy_json_path = output_path / "supplier_compliance_report.json"
    legacy_md_path = output_path / "supplier_compliance_report.md"

    legacy_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    legacy_md_path.write_text(render_markdown(report), encoding="utf-8")

    report["output_files"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "legacy_json": str(legacy_json_path),
        "legacy_markdown": str(legacy_md_path),
    }

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = []

    lines.append("# LocalDocLens Supplier Compliance Report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(f"Supplier: {report['supplier']}")
    lines.append(f"Source File: {report['source_file']}")
    lines.append(f"Overall Risk: {report['overall_risk']}")
    lines.append(f"Decision: {report['decision']}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append("| Finding | Status | Severity | Page | Recommendation |")
    lines.append("|---|---|---|---|---|")

    for item in report["findings"]:
        page = item["page_number"] if item["page_number"] is not None else "N/A"
        lines.append(
            f"| {item['title']} | {item['status']} | {item['severity']} | {page} | {item['recommendation']} |"
        )

    lines.append("")
    lines.append("## Evidence Details")
    lines.append("")

    for item in report["findings"]:
        lines.append(f"### {item['title']}")
        lines.append("")
        lines.append(f"- Status: {item['status']}")
        lines.append(f"- Severity: {item['severity']}")
        lines.append(f"- Details: {item['details']}")

        if item["file_name"] is not None:
            lines.append(f"- Source: {item['file_name']}, page {item['page_number']}")

        if item["evidence_quote"]:
            lines.append("")
            lines.append("Evidence:")
            lines.append("")
            lines.append(f"> {item['evidence_quote']}")

        lines.append("")

    lines.append("## Source Summary")
    lines.append("")
    lines.append(f"- Indexed chunks: {report['source']['indexed_chunks']}")
    lines.append(f"- Files: {', '.join(report['source']['files'])}")
    lines.append(f"- Pages: {', '.join(str(page) for page in report['source']['pages'])}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    result = build_supplier_report()
    print(json.dumps(result, indent=2))
