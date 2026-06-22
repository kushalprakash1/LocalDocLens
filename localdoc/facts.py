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


def clean_value(value: str | None) -> str | None:
    if value is None:
        return None

    value = " ".join(str(value).split())
    value = value.strip(" .,:;-")

    if not value:
        return None

    return value


def normalize_status(value: str | None) -> str | None:
    value = clean_value(value)

    if value is None:
        return None

    return value.lower()


def extract_first(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return clean_value(match.group(1))

    return None


def find_row(rows: list[dict[str, Any]], keywords: list[str]) -> dict[str, Any] | None:
    best = None
    best_score = -1

    for row in rows:
        text_lower = row["text"].lower()
        score = 0

        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += 1

        if score > best_score:
            best_score = score
            best = row

    if best_score <= 0:
        return None

    return best


def value_source(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None

    return {
        "file_name": row["file_name"],
        "page_number": row["page_number"],
        "chunk_id": row["chunk_id"],
        "extraction_method": row["extraction_method"],
        "ocr_confidence": row["ocr_confidence"],
    }


def make_fact(value: Any, row: dict[str, Any] | None, evidence_text: str | None = None) -> dict[str, Any]:
    return {
        "value": value,
        "source": value_source(row),
        "evidence_text": evidence_text if evidence_text is not None else (row["text"] if row else None),
    }


def extract_dates(text: str) -> list[tuple[datetime, str]]:
    dates = []

    for value in re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text):
        try:
            dates.append((datetime.strptime(value, "%m/%d/%Y"), value))
        except Exception:
            pass

    return dates


def latest_date(text: str) -> str | None:
    dates = extract_dates(text)

    if not dates:
        return None

    dates.sort(key=lambda item: item[0])
    return dates[-1][1]


def latest_date_obj(text: str) -> datetime | None:
    dates = extract_dates(text)

    if not dates:
        return None

    dates.sort(key=lambda item: item[0])
    return dates[-1][0]


def extract_supplier_name(joined_text: str) -> str | None:
    patterns = [
        r"Vendor legal name\s*[:\-]?\s*([A-Z][^\n\r]+?)(?:\s+DBA|\s+Primary contact|\s+Email|\s+Phone|\s+Supplier type|\s+Onboarding Checklist|$)",
        r"Supplier legal name\s*[:\-]?\s*([A-Z][^\n\r]+?)(?:\s+DBA|\s+Primary contact|\s+Email|\s+Phone|\s+Supplier type|\s+Bank name|\s+Document type|$)",
        r"Insured supplier\s*[:\-]?\s*([A-Z][^\n\r]+?)(?:\s+Policy carrier|\s+Policy number|\s+Coverage|$)",
        r"Supplier name\s*[:\-]?\s*([A-Z][^\n\r]+?)(?:\s+DBA|\s+Primary contact|\s+Email|\s+Phone|$)",
        r"Vendor name\s*[:\-]?\s*([A-Z][^\n\r]+?)(?:\s+DBA|\s+Primary contact|\s+Email|\s+Phone|$)",
    ]

    result = extract_first(patterns, joined_text)

    if result:
        return result

    company_match = re.search(
        r"([A-Z][A-Za-z0-9 &'.,-]{2,80}\s+(LLC|Inc\.?|Corporation|Corp\.?|Ltd\.?|Limited|Co\.?|Company))",
        joined_text[:2500],
    )

    if company_match:
        return clean_value(company_match.group(1))

    return None


def status_from_text(text: str, positive_terms: list[str], negative_terms: list[str]) -> str:
    text_lower = text.lower()

    has_negative = any(term.lower() in text_lower for term in negative_terms)
    has_positive = any(term.lower() in text_lower for term in positive_terms)

    if has_negative and not has_positive:
        return "failed"

    if has_positive and not has_negative:
        return "passed"

    if has_positive and has_negative:
        return "review"

    return "unknown"


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
        return rows

    target = Path(file_name).name.lower()

    selected = [
        row
        for row in rows
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


def group_rows_by_file(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {}

    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)

    return grouped


def extract_facts_for_file(file_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (row["page_number"], row["chunk_id"]))
    joined_text = "\n".join(row["text"] for row in rows)

    summary_row = find_row(rows, ["supplier summary", "vendor legal name", "supplier legal name", "primary contact"])
    insurance_row = find_row(rows, ["certificate of insurance", "policy number", "policy expiration date", "coi"])
    bank_row = find_row(rows, ["bank verification", "bank name", "routing number", "ach"])
    agreement_row = find_row(rows, ["supplier agreement", "signature", "signed by"])
    food_safety_row = find_row(rows, ["food safety", "supplemental compliance", "sanctions screening"])

    supplier_name = extract_supplier_name(joined_text)

    dba_name = extract_first(
        [
            r"DBA name\s*[:\-]?\s*([^\n\r]+?)(?:\s+Primary contact|\s+Email|\s+Phone|\s+Supplier type|$)",
        ],
        joined_text,
    )

    primary_contact = extract_first(
        [
            r"Primary contact\s*[:\-]?\s*([^\n\r]+?)(?:\s+Email|\s+Phone|\s+Supplier type|$)",
            r"Primary contact\s+([A-Z][^\n\r]+?)(?:\s+Email|\s+Phone|\s+Supplier type|$)",
        ],
        joined_text,
    )

    email = extract_first(
        [
            r"Email\s*[:\-]?\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
            r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b",
        ],
        joined_text,
    )

    phone = extract_first(
        [
            r"Phone\s*[:\-]?\s*(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})",
        ],
        joined_text,
    )

    supplier_type = extract_first(
        [
            r"Supplier type\s*[:\-]?\s*([^\n\r]+?)(?:\s+Onboarding Checklist|\s+Requirement|$)",
        ],
        joined_text,
    )

    policy_carrier = extract_first(
        [
            r"Policy carrier\s*[:\-]?\s*([^\n\r]+?)(?:\s+Policy number|\s+Coverage|$)",
        ],
        joined_text,
    )

    policy_number = extract_first(
        [
            r"Policy number\s*[:\-]?\s*([A-Z0-9\-]+)",
        ],
        joined_text,
    )

    insurance_expiration = latest_date(insurance_row["text"] if insurance_row else joined_text)
    insurance_latest_date_obj = latest_date_obj(insurance_row["text"] if insurance_row else joined_text)

    insurance_status = "unknown"

    if insurance_row:
        insurance_text_lower = insurance_row["text"].lower()

        if any(term in insurance_text_lower for term in ["not expired", "is active", "appears active", "coverage is valid", "valid for supplier onboarding"]):
            insurance_status = "active"

        if any(term in insurance_text_lower for term in ["appears expired", "is expired", "expired because", "coverage expired"]):
            insurance_status = "expired"

        if insurance_latest_date_obj and insurance_latest_date_obj.date() < datetime.utcnow().date() and insurance_status != "active":
            insurance_status = "expired"

        if insurance_latest_date_obj and insurance_latest_date_obj.date() >= datetime.utcnow().date() and insurance_status == "unknown":
            insurance_status = "active"

    w9_status = status_from_text(
        joined_text,
        positive_terms=[
            "w-9 tax form received",
            "signed w-9 received",
            "w-9 received",
            "received signed w-9",
        ],
        negative_terms=[
            "w-9 tax form missing",
            "w-9 missing",
            "missing w-9",
            "not received",
            "must provide signed w-9",
        ],
    )

    agreement_status = status_from_text(
        agreement_row["text"] if agreement_row else joined_text,
        positive_terms=[
            "agreement status: signed",
            "supplier representative signature signed",
            "includes the required supplier signature",
            "signed by both parties",
            "both parties signed",
            "agreement is complete",
        ],
        negative_terms=[
            "supplier representative signature missing",
            "missing the supplier signature",
            "missing supplier signature",
            "pending signature",
        ],
    )

    bank_verification_status = "unknown"

    if bank_row:
        bank_text_lower = bank_row["text"].lower()

        if "verified" in bank_text_lower:
            bank_verification_status = "verified"

    bank_name = extract_first(
        [
            r"Bank name\s*[:\-]?\s*([^\n\r]+?)(?:\s+Routing number|\s+Account number|\s+ACH|$)",
        ],
        joined_text,
    )

    routing_number = extract_first(
        [
            r"Routing number\s*[:\-]?\s*([0-9]+)",
        ],
        joined_text,
    )

    account_ending = extract_first(
        [
            r"Account number\s*[:\-]?\s*ending in\s*([0-9]+)",
            r"account ending\s*([0-9]+)",
        ],
        joined_text,
    )

    payment_terms = extract_first(
        [
            r"Payment terms\s*[:\-]?\s*(Net\s*\d+)",
            r"Payment terms\s+(Net\s*\d+)",
        ],
        joined_text,
    )

    remittance_email = extract_first(
        [
            r"Remittance email\s*[:\-]?\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
        ],
        joined_text,
    )

    buyer_signer = extract_first(
        [
            r"Buyer representative signature\s*[:\-]?\s*Signed\s+Signed by\s+([A-Z][A-Za-z .'-]+?)\s+on\s+\d{2}/\d{2}/\d{4}",
            r"Signed by\s+([A-Z][A-Za-z .'-]+?)\s+on\s+\d{2}/\d{2}/\d{4}",
        ],
        agreement_row["text"] if agreement_row else joined_text,
    )

    supplier_signer = extract_first(
        [
            r"Supplier representative signature\s*[:\-]?\s*Signed\s+Signed by\s+([A-Z][A-Za-z .'-]+?)\s+on\s+\d{2}/\d{2}/\d{4}",
        ],
        agreement_row["text"] if agreement_row else joined_text,
    )

    agreement_effective_date = extract_first(
        [
            r"Agreement effective date\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
            r"both parties signed on\s*(\d{2}/\d{2}/\d{4})",
        ],
        agreement_row["text"] if agreement_row else joined_text,
    )

    food_safety_status = "unknown"

    if food_safety_row:
        food_text_lower = food_safety_row["text"].lower()

        if "food safety certificate: received" in food_text_lower or "food safety certificate received" in food_text_lower:
            food_safety_status = "received"

        if "certificate review status: passed" in food_text_lower:
            food_safety_status = "passed"

    sanctions_status = extract_first(
        [
            r"Sanctions screening\s*[:\-]?\s*([^\n\r]+?)(?:\s+Conflict|\s+Overall|$)",
        ],
        joined_text,
    )

    conflict_status = extract_first(
        [
            r"Conflict of interest disclosure\s*[:\-]?\s*([^\n\r]+?)(?:\s+Overall|$)",
        ],
        joined_text,
    )

    failed_items = []

    if insurance_status == "expired":
        failed_items.append("expired insurance")

    if w9_status == "failed":
        failed_items.append("missing W-9")

    if agreement_status == "failed":
        failed_items.append("missing supplier agreement signature")

    review_items = []

    if insurance_status == "unknown":
        review_items.append("unclear insurance status")

    if w9_status in {"unknown", "review"}:
        review_items.append("unclear W-9 status")

    if agreement_status in {"unknown", "review"}:
        review_items.append("unclear agreement signature status")

    if bank_verification_status == "unknown":
        review_items.append("unclear bank verification status")

    if failed_items:
        overall_risk = "High"
        decision = "Do not approve until failed compliance items are resolved."
    elif review_items:
        overall_risk = "Medium"
        decision = "Hold approval until unclear compliance items are reviewed."
    else:
        overall_risk = "Low"
        decision = "Supplier appears ready for approval based on available evidence."

    facts = {
        "generated_at": utc_now(),
        "file_name": file_name,
        "supplier_name": make_fact(supplier_name, summary_row),
        "dba_name": make_fact(dba_name, summary_row),
        "primary_contact": make_fact(primary_contact, summary_row),
        "email": make_fact(email, summary_row),
        "phone": make_fact(phone, summary_row),
        "supplier_type": make_fact(supplier_type, summary_row),
        "w9_status": make_fact(w9_status, summary_row),
        "insurance": {
            "status": make_fact(insurance_status, insurance_row),
            "policy_carrier": make_fact(policy_carrier, insurance_row),
            "policy_number": make_fact(policy_number, insurance_row),
            "latest_expiration_date": make_fact(insurance_expiration, insurance_row),
        },
        "bank": {
            "verification_status": make_fact(bank_verification_status, bank_row),
            "bank_name": make_fact(bank_name, bank_row),
            "routing_number": make_fact(routing_number, bank_row),
            "account_ending": make_fact(account_ending, bank_row),
            "payment_terms": make_fact(payment_terms, bank_row),
            "remittance_email": make_fact(remittance_email, bank_row),
        },
        "agreement": {
            "status": make_fact(agreement_status, agreement_row),
            "buyer_signer": make_fact(buyer_signer, agreement_row),
            "supplier_signer": make_fact(supplier_signer, agreement_row),
            "effective_date": make_fact(agreement_effective_date, agreement_row),
        },
        "additional_compliance": {
            "food_safety_certificate": make_fact(food_safety_status, food_safety_row),
            "sanctions_screening": make_fact(sanctions_status, food_safety_row),
            "conflict_of_interest": make_fact(conflict_status, food_safety_row),
        },
        "risk": {
            "overall_risk": overall_risk,
            "decision": decision,
            "failed_items": failed_items,
            "review_items": review_items,
        },
        "source": {
            "pages": sorted({row["page_number"] for row in rows}),
            "chunks": len(rows),
            "extraction_methods": sorted({row["extraction_method"] for row in rows if row["extraction_method"]}),
            "avg_ocr_confidence": average_ocr_confidence(rows),
        },
    }

    return facts


def average_ocr_confidence(rows: list[dict[str, Any]]) -> float | None:
    values = [
        row["ocr_confidence"]
        for row in rows
        if row["ocr_confidence"] is not None
    ]

    if not values:
        return None

    return round(sum(values) / len(values), 4)


def render_facts_markdown(all_facts: list[dict[str, Any]]) -> str:
    lines = []

    lines.append("# LocalDocLens Extracted Supplier Facts")
    lines.append("")

    for facts in all_facts:
        lines.append(f"## {facts['supplier_name']['value'] or facts['file_name']}")
        lines.append("")
        lines.append(f"- File: {facts['file_name']}")
        lines.append(f"- Overall risk: {facts['risk']['overall_risk']}")
        lines.append(f"- Decision: {facts['risk']['decision']}")
        lines.append(f"- Email: {facts['email']['value']}")
        lines.append(f"- Phone: {facts['phone']['value']}")
        lines.append(f"- W-9 status: {facts['w9_status']['value']}")
        lines.append(f"- Insurance status: {facts['insurance']['status']['value']}")
        lines.append(f"- Policy number: {facts['insurance']['policy_number']['value']}")
        lines.append(f"- Insurance expiration: {facts['insurance']['latest_expiration_date']['value']}")
        lines.append(f"- Bank name: {facts['bank']['bank_name']['value']}")
        lines.append(f"- Bank verification: {facts['bank']['verification_status']['value']}")
        lines.append(f"- Payment terms: {facts['bank']['payment_terms']['value']}")
        lines.append(f"- Agreement status: {facts['agreement']['status']['value']}")
        lines.append(f"- Buyer signer: {facts['agreement']['buyer_signer']['value']}")
        lines.append(f"- Supplier signer: {facts['agreement']['supplier_signer']['value']}")
        lines.append(f"- Food safety certificate: {facts['additional_compliance']['food_safety_certificate']['value']}")
        lines.append("")

    return "\n".join(lines)


def extract_supplier_facts(file_name: str | None = None, output_dir: str = "artifacts") -> dict[str, Any]:
    rows = load_rows()
    selected = filter_rows_by_file(rows, file_name)
    grouped = group_rows_by_file(selected)

    all_facts = []

    for grouped_file_name, grouped_rows in sorted(grouped.items()):
        all_facts.append(extract_facts_for_file(grouped_file_name, grouped_rows))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combined_json_path = output_path / "extracted_supplier_facts.json"
    combined_md_path = output_path / "extracted_supplier_facts.md"

    result = {
        "generated_at": utc_now(),
        "file_filter": file_name,
        "num_files": len(all_facts),
        "facts": all_facts,
        "output_files": {
            "combined_json": str(combined_json_path),
            "combined_markdown": str(combined_md_path),
            "per_file_json": [],
        },
    }

    combined_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    combined_md_path.write_text(render_facts_markdown(all_facts), encoding="utf-8")

    for facts in all_facts:
        safe_stem = Path(facts["file_name"]).stem.replace(" ", "_")
        per_file_path = output_path / f"{safe_stem}_supplier_facts.json"
        per_file_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
        result["output_files"]["per_file_json"].append(str(per_file_path))

    combined_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    print(json.dumps(extract_supplier_facts(), indent=2))
