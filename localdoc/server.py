import hashlib
import json
import math
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
from fastapi import FastAPI
from pydantic import BaseModel, validator
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from localdoc.runtime_security import install_cors, install_runtime_security
from localdoc.config import EMBEDDING_MODEL_NAME, LANCEDB_PATH
from localdoc.facts import extract_facts_for_file
from localdoc.memory import SupplierMemoryManager
from localdoc.learning import LearningExampleStore
from localdoc.verifier import verify_answer


CACHE_VERSION = "hybrid-rag-v1"
CACHE_DB_PATH = Path("data/db/answer_cache.sqlite")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"


class AskRequest(BaseModel):
    question: str
    top_k: int = 6
    file: str = ""
    mode: str = "auto"
    use_cache: bool = True
    require_verified: bool = False
    use_llm: bool = True



    @validator("file")
    def validate_file_filter(cls, value):
        from localdoc.runtime_security import validate_safe_file_name

        value = (value or "").strip()

        if not value:
            return ""

        return validate_safe_file_name(value)

class AskResponse(BaseModel):
    question: str
    answer: str
    latency_s: float
    retrieval_latency_s: float
    generation_latency_s: float
    evidence: list[dict[str, Any]]
    cache_status: str
    cache_key: str
    cache_verified: bool
    answer_mode: str
    file_filter: str | None


class VerifyCacheRequest(BaseModel):
    cache_key: str
    verified_by: str = "local_reviewer"
    note: str = ""


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9$./-]+", text.lower())


def normalize_question(question: str) -> str:
    question = question.lower().strip()
    question = re.sub(r"\s+", " ", question)
    question = question.rstrip("?!. ")
    return question


def normalize_file_name(file_name: str) -> str:
    if not file_name:
        return ""
    return Path(file_name).name.lower().strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_text(text: str, max_chars: int = 900) -> str:
    text = " ".join(str(text).split())
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def safe_float(value):
    try:
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def extract_supplier(text: str) -> str:
    patterns = [
        r"Supplier legal name\s*[:\-]?\s*([^\n\r]+)",
        r"Vendor legal name\s*[:\-]?\s*([^\n\r]+)",
        r"Insured supplier\s*[:\-]?\s*([^\n\r]+)",
        r"Supplier name\s*[:\-]?\s*([^\n\r]+)",
        r"Legal name\s*[:\-]?\s*([^\n\r]+)",
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
        "Status",
        "Notes",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        value = match.group(1).strip()

        for stop_word in stop_words:
            value = re.split(
                r"\s+" + re.escape(stop_word) + r"\b",
                value,
                flags=re.IGNORECASE,
            )[0].strip()

        value = value.strip(" .:-")

        if value and len(value) >= 2:
            return value

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


def is_compliance_question(question: str) -> bool:
    q = question.lower()

    compliance_terms = [
        "insurance",
        "expired",
        "coi",
        "certificate of insurance",
        "w-9",
        "w9",
        "signature",
        "signed",
        "supplier agreement",
        "bank verification",
        "ach",
        "compliance status",
        "approval",
        "approve",
        "risk",
        "onboarding decision",
    ]

    return any(term in q for term in compliance_terms)


def score_row(row: dict[str, Any], required_any: list[str], preferred: list[str]) -> float:
    text_lower = row["text"].lower()

    if required_any and not any(term.lower() in text_lower for term in required_any):
        return -1.0

    score = 0.0

    for term in preferred:
        if term.lower() in text_lower:
            score += 1.0

    score += min(len(row["text"]) / 5000, 0.25)

    return score


def find_best_row(rows: list[dict[str, Any]], required_any: list[str], preferred: list[str]) -> dict[str, Any] | None:
    best_row = None
    best_score = -1.0

    for row in rows:
        score = score_row(row, required_any, preferred)

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
        "text": row["text"] if row else None,
        "evidence_quote": clean_text(row["text"]) if row else None,
        "recommendation": recommendation,
        "details": details,
    }


def detect_insurance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = find_best_row(
        rows,
        required_any=["insurance", "certificate of insurance", "coi"],
        preferred=[
            "certificate of insurance",
            "policy expiration date",
            "appears expired",
            "not expired",
            "active",
            "valid",
            "additional insured",
        ],
    )

    status = "unknown"
    severity = "medium"
    details = "Insurance status could not be fully determined."
    recommendation = "Review insurance coverage before supplier approval."

    if row:
        text_lower = row["text"].lower()
        latest_text = latest_date_text(row["text"])
        parsed_dates = extract_dates(row["text"])
        latest_parsed_date = max(parsed_dates) if parsed_dates else None

        positive_active = any(
            phrase in text_lower
            for phrase in [
                "not expired",
                "is active",
                "appears active",
                "coverage is valid",
                "insurance coverage is valid",
                "valid for supplier onboarding",
            ]
        )

        explicit_expired = any(
            phrase in text_lower
            for phrase in [
                "appears expired",
                "is expired",
                "expired because",
                "coverage expired",
            ]
        )

        date_expired = False

        if latest_parsed_date and latest_parsed_date.date() < datetime.utcnow().date():
            date_expired = True

        if explicit_expired or (date_expired and not positive_active):
            status = "failed"
            severity = "high"
            details = f"Certificate of Insurance appears expired. Latest detected policy date: {latest_text}."
            recommendation = "Request an updated Certificate of Insurance before approval."
        elif positive_active or (latest_parsed_date and latest_parsed_date.date() >= datetime.utcnow().date()):
            status = "passed"
            severity = "low"
            details = f"Certificate of Insurance appears active. Latest detected policy date: {latest_text}."
            recommendation = "No immediate action required for insurance."
        else:
            status = "review"
            severity = "medium"
            details = f"Certificate of Insurance found, but status is unclear. Latest detected policy date: {latest_text}."
            recommendation = "Review Certificate of Insurance before approval."

    return make_finding(
        finding_id="insurance",
        title="Insurance Coverage",
        status=status,
        severity=severity,
        row=row,
        recommendation=recommendation,
        details=details,
    )


def detect_w9(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = find_best_row(
        rows,
        required_any=["w-9", "w9"],
        preferred=[
            "missing",
            "not received",
            "received",
            "signed w-9",
            "reviewed",
            "before approval",
        ],
    )

    status = "unknown"
    severity = "medium"
    details = "W-9 status could not be fully determined."
    recommendation = "Review W-9 status before supplier approval."

    if row:
        text_lower = row["text"].lower()

        negative = any(
            phrase in text_lower
            for phrase in [
                "w-9 tax form missing",
                "w-9 missing",
                "missing w-9",
                "not received",
                "must provide signed w-9",
            ]
        )

        positive = any(
            phrase in text_lower
            for phrase in [
                "w-9 tax form received",
                "signed w-9 received",
                "w-9 received",
                "received signed w-9",
            ]
        )

        if negative:
            status = "failed"
            severity = "high"
            details = "W-9 tax form is missing or has not been received."
            recommendation = "Collect a signed W-9 before supplier approval."
        elif positive:
            status = "passed"
            severity = "low"
            details = "W-9 tax form appears received and reviewed."
            recommendation = "No immediate action required for W-9."
        else:
            status = "review"
            severity = "medium"
            details = "W-9 evidence exists, but status is unclear."
            recommendation = "Review W-9 evidence before approval."

    return make_finding(
        finding_id="w9",
        title="W-9 Tax Form",
        status=status,
        severity=severity,
        row=row,
        recommendation=recommendation,
        details=details,
    )


def detect_signature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = find_best_row(
        rows,
        required_any=["supplier agreement", "signature"],
        preferred=[
            "supplier agreement",
            "supplier representative signature",
            "missing",
            "signed",
            "both parties",
            "pending signature",
        ],
    )

    status = "unknown"
    severity = "medium"
    details = "Supplier agreement signature status could not be fully determined."
    recommendation = "Review supplier agreement signature before approval."

    if row:
        text_lower = row["text"].lower()

        negative = any(
            phrase in text_lower
            for phrase in [
                "supplier representative signature missing",
                "missing the supplier signature",
                "missing supplier signature",
                "pending signature",
            ]
        )

        positive = any(
            phrase in text_lower
            for phrase in [
                "agreement status: signed",
                "supplier representative signature signed",
                "includes the required supplier signature",
                "signed by both parties",
                "both parties signed",
                "agreement is complete",
            ]
        )

        if negative and not positive:
            status = "failed"
            severity = "high"
            details = "Supplier agreement is missing the supplier signature."
            recommendation = "Collect the supplier representative signature before approval."
        elif positive:
            status = "passed"
            severity = "low"
            details = "Supplier agreement appears signed by the required party."
            recommendation = "No immediate action required for supplier agreement signature."
        else:
            status = "review"
            severity = "medium"
            details = "Supplier agreement signature evidence exists, but status is unclear."
            recommendation = "Review supplier agreement signature before approval."

    return make_finding(
        finding_id="supplier_signature",
        title="Supplier Agreement Signature",
        status=status,
        severity=severity,
        row=row,
        recommendation=recommendation,
        details=details,
    )


def detect_bank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = find_best_row(
        rows,
        required_any=["bank verification", "ach"],
        preferred=[
            "verified",
            "bank letter",
            "ach",
            "no bank account change",
        ],
    )

    status = "unknown"
    severity = "medium"
    details = "Bank verification status could not be fully determined."
    recommendation = "Review bank verification before payment setup."

    if row:
        text_lower = row["text"].lower()

        if "verified" in text_lower:
            status = "passed"
            severity = "low"
            details = "Bank verification is marked as verified."
            recommendation = "No immediate action required for bank verification."
        else:
            status = "review"
            severity = "medium"
            details = "Bank verification evidence exists but is not clearly verified."
            recommendation = "Review bank verification before payment setup."

    return make_finding(
        finding_id="bank_verification",
        title="Bank Verification",
        status=status,
        severity=severity,
        row=row,
        recommendation=recommendation,
        details=details,
    )


def analyze_supplier_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    joined_text = "\n".join(row["text"] for row in rows)
    supplier = extract_supplier(joined_text)

    findings = [
        detect_insurance(rows),
        detect_w9(rows),
        detect_signature(rows),
        detect_bank(rows),
    ]

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

    return {
        "supplier": supplier,
        "overall_risk": overall_risk,
        "decision": decision,
        "findings": findings,
    }


def make_compliance_answer(question: str, rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    q = question.lower()
    analysis = analyze_supplier_rows(rows)
    supplier = analysis["supplier"]
    findings = analysis["findings"]

    def get_finding(finding_id: str) -> dict[str, Any]:
        for item in findings:
            if item["id"] == finding_id:
                return item
        return findings[0]

    if "insurance" in q or "expired" in q or "coi" in q:
        finding = get_finding("insurance")
        evidence = [finding] if finding.get("file_name") else []

        if finding["status"] == "failed":
            answer = f"Direct answer: {supplier} has an insurance issue.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        elif finding["status"] == "passed":
            answer = f"Direct answer: {supplier} does not appear to have expired insurance.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        else:
            answer = f"Direct answer: Insurance status for {supplier} needs review.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."

        return answer, evidence

    if "w-9" in q or "w9" in q:
        finding = get_finding("w9")
        evidence = [finding] if finding.get("file_name") else []

        if finding["status"] == "failed":
            answer = f"Direct answer: {supplier} is missing a W-9.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        elif finding["status"] == "passed":
            answer = f"Direct answer: {supplier} appears to have a received W-9.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        else:
            answer = f"Direct answer: W-9 status for {supplier} needs review.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."

        return answer, evidence

    if "signature" in q or "signed" in q or "agreement" in q:
        finding = get_finding("supplier_signature")
        evidence = [finding] if finding.get("file_name") else []

        if finding["status"] == "failed":
            answer = f"Direct answer: {supplier} is missing the supplier agreement signature.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        elif finding["status"] == "passed":
            answer = f"Direct answer: {supplier}'s supplier agreement appears signed.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        else:
            answer = f"Direct answer: Supplier agreement signature status for {supplier} needs review.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."

        return answer, evidence

    if "bank" in q or "ach" in q:
        finding = get_finding("bank_verification")
        evidence = [finding] if finding.get("file_name") else []

        if finding["status"] == "passed":
            answer = f"Direct answer: The bank verification status for {supplier} is verified.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."
        else:
            answer = f"Direct answer: Bank verification status for {supplier} needs review.\n\n{finding['details']}\n\nEvidence: {finding['file_name']}, page {finding['page_number']}."

        return answer, evidence

    summary_lines = []

    for finding in findings:
        summary_lines.append(
            f"- {finding['title']}: {finding['status']} ({finding['details']})"
        )

    evidence = [
        finding
        for finding in findings
        if finding.get("file_name")
    ]

    answer = (
        f"Direct answer: {supplier} overall risk is {analysis['overall_risk']}.\n\n"
        f"Decision: {analysis['decision']}\n\n"
        + "\n".join(summary_lines)
    )

    return answer, evidence


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*thinking\s*[:\-].*$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return text


def call_ollama(prompt: str) -> str:
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

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return strip_thinking(result.get("response", "").strip())


def make_general_rag_prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    blocks = []

    for index, item in enumerate(evidence, start=1):
        blocks.append(
            f"[E{index}] File: {item['file_name']} | Page: {item['page_number']}\n"
            f"{clean_text(item['text'], max_chars=1800)}"
        )

    context = "\n\n".join(blocks)

    return f"""
You are LocalDocLens, a local supplier-document RAG assistant.

Answer the user's question using ONLY the evidence below.
Do not use outside knowledge.
If the evidence does not contain the answer, say: "I cannot determine that from the provided document evidence."
Be direct and concise.
Always include the source file and page number in the answer.

Evidence:
{context}

Question:
{question}

Answer:
""".strip()


def make_extract_fallback_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "I could not find relevant document evidence for that question."

    top = evidence[0]

    return (
        "I found relevant evidence, but the local LLM fallback was unavailable.\n\n"
        f"Most relevant source: {top['file_name']}, page {top['page_number']}.\n\n"
        f"Evidence excerpt: {clean_text(top['text'], max_chars=1000)}"
    )


def make_general_rag_answer(question: str, evidence: list[dict[str, Any]], use_llm: bool) -> str:
    if not evidence:
        return "I could not find relevant document evidence for that question."

    if not use_llm:
        return make_extract_fallback_answer(question, evidence)

    prompt = make_general_rag_prompt(question, evidence)

    try:
        answer = call_ollama(prompt)
    except urllib.error.URLError:
        return make_extract_fallback_answer(question, evidence)
    except TimeoutError:
        return make_extract_fallback_answer(question, evidence)
    except Exception as exc:
        return (
            "I found relevant evidence, but local LLM generation failed.\n\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            + make_extract_fallback_answer(question, evidence)
        )

    if not answer:
        return make_extract_fallback_answer(question, evidence)

    return answer


def is_fact_question(question: str) -> bool:
    q = question.lower()

    fact_terms = [
        "email",
        "primary contact",
        "contact",
        "phone",
        "bank name",
        "routing",
        "account",
        "payment terms",
        "remittance",
        "policy number",
        "policy carrier",
        "expiration date",
        "food safety",
        "sanctions",
        "conflict of interest",
        "who signed",
        "signed by",
        "supplier type",
        "dba",
    ]

    return any(term in q for term in fact_terms)


def get_nested_fact(facts: dict[str, Any], path: list[str]) -> dict[str, Any] | None:
    current = facts

    for part in path:
        if not isinstance(current, dict):
            return None

        if part not in current:
            return None

        current = current[part]

    if isinstance(current, dict) and "value" in current:
        return current

    return None


def fact_to_evidence(fact: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not fact:
        return []

    source = fact.get("source")

    if not source:
        return []

    return [
        {
            "chunk_id": source.get("chunk_id"),
            "file_name": source.get("file_name"),
            "page_number": source.get("page_number"),
            "text": fact.get("evidence_text"),
            "extraction_method": source.get("extraction_method"),
            "ocr_confidence": source.get("ocr_confidence"),
            "score": 1.0,
        }
    ]


def format_fact_source(fact: dict[str, Any] | None) -> str:
    if not fact or not fact.get("source"):
        return "Source: unavailable"

    source = fact["source"]

    return f"Source: {source.get('file_name')}, page {source.get('page_number')}"


def make_fact_answer(question: str, rows: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    if not rows:
        return None, []

    q = question.lower()
    source_file = sorted({row["file_name"] for row in rows})[0]
    try:
        facts = runtime.memory.get_facts(source_file, rows)
    except NameError:
        facts = extract_facts_for_file(source_file, rows)

    if facts is None:
        facts = extract_facts_for_file(source_file, rows)

    supplier_fact = get_nested_fact(facts, ["supplier_name"])
    supplier = supplier_fact["value"] if supplier_fact and supplier_fact.get("value") else source_file

    if "email" in q:
        fact = get_nested_fact(facts, ["email"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "primary contact" in q or ("contact" in q and "email" not in q):
        fact = get_nested_fact(facts, ["primary_contact"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "phone" in q:
        fact = get_nested_fact(facts, ["phone"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "payment terms" in q or "payment term" in q:
        fact = get_nested_fact(facts, ["bank", "payment_terms"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "bank name" in q or q.strip() == "what is the bank name":
        fact = get_nested_fact(facts, ["bank", "bank_name"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "routing" in q:
        fact = get_nested_fact(facts, ["bank", "routing_number"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "account" in q and "ending" in q:
        fact = get_nested_fact(facts, ["bank", "account_ending"])
        if fact and fact.get("value"):
            answer = f"Direct answer: account ending in {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "policy number" in q:
        fact = get_nested_fact(facts, ["insurance", "policy_number"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "policy carrier" in q or "insurance carrier" in q:
        fact = get_nested_fact(facts, ["insurance", "policy_carrier"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "expiration" in q and ("insurance" in q or "policy" in q):
        fact = get_nested_fact(facts, ["insurance", "latest_expiration_date"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "food safety" in q or "safety certification" in q or "safety certificate" in q:
        fact = get_nested_fact(facts, ["additional_compliance", "food_safety_certificate"])
        if fact and fact.get("value"):
            value = str(fact["value"]).lower()

            if value in {"received", "passed"}:
                answer = f"Direct answer: Yes. Food safety certificate status is {fact['value']}.\n\n{format_fact_source(fact)}."
            else:
                answer = f"Direct answer: Food safety certificate status is {fact['value']}.\n\n{format_fact_source(fact)}."

            return answer, fact_to_evidence(fact)

    if "sanctions" in q:
        fact = get_nested_fact(facts, ["additional_compliance", "sanctions_screening"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "conflict" in q:
        fact = get_nested_fact(facts, ["additional_compliance", "conflict_of_interest"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "who signed" in q or "signed by" in q:
        buyer_fact = get_nested_fact(facts, ["agreement", "buyer_signer"])
        supplier_signer_fact = get_nested_fact(facts, ["agreement", "supplier_signer"])

        buyer = buyer_fact["value"] if buyer_fact and buyer_fact.get("value") else None
        supplier_signer = supplier_signer_fact["value"] if supplier_signer_fact and supplier_signer_fact.get("value") else None

        if buyer or supplier_signer:
            pieces = []

            if buyer:
                pieces.append(f"buyer signer: {buyer}")

            if supplier_signer:
                pieces.append(f"supplier signer: {supplier_signer}")

            evidence = []

            if buyer_fact:
                evidence.extend(fact_to_evidence(buyer_fact))

            if supplier_signer_fact:
                evidence.extend(fact_to_evidence(supplier_signer_fact))

            answer = f"Direct answer: {', '.join(pieces)}.\n\nSource: {source_file}, page {evidence[0]['page_number'] if evidence else 'unknown'}."
            return answer, evidence

    if "supplier type" in q:
        fact = get_nested_fact(facts, ["supplier_type"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "dba" in q:
        fact = get_nested_fact(facts, ["dba_name"])
        if fact and fact.get("value"):
            answer = f"Direct answer: {fact['value']}\n\n{format_fact_source(fact)}."
            return answer, fact_to_evidence(fact)

    if "risk" in q or "decision" in q:
        risk = facts["risk"]["overall_risk"]
        decision = facts["risk"]["decision"]
        answer = f"Direct answer: {supplier} overall risk is {risk}.\n\nDecision: {decision}."
        evidence = fact_to_evidence(supplier_fact)
        return answer, evidence

    return None, []


class AnswerCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_version TEXT NOT NULL,
                    doc_fingerprint TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    answer TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    verified_by TEXT,
                    verified_at TEXT,
                    note TEXT,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_answer_cache_question
                ON answer_cache (normalized_question)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_answer_cache_verified
                ON answer_cache (verified)
                """
            )

    def make_key(
        self,
        doc_fingerprint: str,
        question: str,
        top_k: int,
        file_filter: str | None,
        answer_mode: str,
        use_llm: bool,
    ) -> str:
        normalized = normalize_question(question)

        raw = json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "doc_fingerprint": doc_fingerprint,
                "normalized_question": normalized,
                "top_k": top_k,
                "file_filter": file_filter or "",
                "answer_mode": answer_mode,
                "use_llm": use_llm,
            },
            sort_keys=True,
        )

        return sha256_text(raw)

    def get(self, cache_key: str, require_verified: bool) -> dict[str, Any] | None:
        with self.connect() as conn:
            if require_verified:
                row = conn.execute(
                    """
                    SELECT cache_key, answer, evidence_json, verified, hit_count
                    FROM answer_cache
                    WHERE cache_key = ? AND verified = 1
                    """,
                    (cache_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT cache_key, answer, evidence_json, verified, hit_count
                    FROM answer_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()

            if not row:
                return None

            conn.execute(
                """
                UPDATE answer_cache
                SET hit_count = hit_count + 1,
                    updated_at = ?
                WHERE cache_key = ?
                """,
                (utc_now(), cache_key),
            )

            return {
                "cache_key": row[0],
                "answer": row[1],
                "evidence": json.loads(row[2]),
                "verified": bool(row[3]),
                "hit_count": int(row[4]) + 1,
            }

    def set(
        self,
        cache_key: str,
        doc_fingerprint: str,
        question: str,
        top_k: int,
        answer: str,
        evidence: list[dict[str, Any]],
    ):
        now = utc_now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO answer_cache (
                    cache_key,
                    cache_version,
                    doc_fingerprint,
                    normalized_question,
                    original_question,
                    top_k,
                    answer,
                    evidence_json,
                    verified,
                    verified_by,
                    verified_at,
                    note,
                    hit_count,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT verified FROM answer_cache WHERE cache_key = ?), 0),
                        COALESCE((SELECT verified_by FROM answer_cache WHERE cache_key = ?), NULL),
                        COALESCE((SELECT verified_at FROM answer_cache WHERE cache_key = ?), NULL),
                        COALESCE((SELECT note FROM answer_cache WHERE cache_key = ?), NULL),
                        COALESCE((SELECT hit_count FROM answer_cache WHERE cache_key = ?), 0),
                        COALESCE((SELECT created_at FROM answer_cache WHERE cache_key = ?), ?),
                        ?)
                """,
                (
                    cache_key,
                    CACHE_VERSION,
                    doc_fingerprint,
                    normalize_question(question),
                    question,
                    top_k,
                    answer,
                    json.dumps(evidence, ensure_ascii=False),
                    cache_key,
                    cache_key,
                    cache_key,
                    cache_key,
                    cache_key,
                    cache_key,
                    now,
                    now,
                ),
            )

    def verify(self, cache_key: str, verified_by: str, note: str) -> bool:
        with self.connect() as conn:
            result = conn.execute(
                """
                UPDATE answer_cache
                SET verified = 1,
                    verified_by = ?,
                    verified_at = ?,
                    note = ?,
                    updated_at = ?
                WHERE cache_key = ?
                """,
                (verified_by, utc_now(), note, utc_now(), cache_key),
            )

            return result.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
            verified = conn.execute("SELECT COUNT(*) FROM answer_cache WHERE verified = 1").fetchone()[0]
            hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM answer_cache").fetchone()[0]

            rows = conn.execute(
                """
                SELECT cache_key, normalized_question, verified, hit_count, created_at, updated_at
                FROM answer_cache
                ORDER BY updated_at DESC
                LIMIT 20
                """
            ).fetchall()

            recent = []

            for row in rows:
                recent.append(
                    {
                        "cache_key": row[0],
                        "normalized_question": row[1],
                        "verified": bool(row[2]),
                        "hit_count": int(row[3]),
                        "created_at": row[4],
                        "updated_at": row[5],
                    }
                )

            return {
                "cache_db_path": str(self.db_path),
                "total_cached_answers": int(total),
                "verified_cached_answers": int(verified),
                "total_cache_hits": int(hits),
                "recent": recent,
            }


class WarmRuntime:
    def __init__(self):
        print("Loading LocalDocLens warm runtime...")

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.db = lancedb.connect(LANCEDB_PATH)
        self.table = self.db.open_table("chunks")

        self.df = self.table.to_pandas()
        self.df["text"] = self.df["text"].fillna("").astype(str)

        self.rows = []

        for _, row in self.df.iterrows():
            self.rows.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "file_name": str(row["file_name"]),
                    "page_number": int(row["page_number"]),
                    "text": str(row["text"]),
                    "extraction_method": str(row.get("extraction_method", "")),
                    "ocr_confidence": safe_float(row.get("ocr_confidence")),
                }
            )

        self.bm25 = BM25Okapi([tokenize(row["text"]) for row in self.rows])
        self.cache = AnswerCache(CACHE_DB_PATH)
        self.memory = SupplierMemoryManager()
        self.learning = LearningExampleStore()
        self.memory_refresh_stats = self.memory.refresh(self.rows)

        print(f"Warm runtime ready. Chunks loaded: {len(self.rows)}")
        print(f"Files indexed: {sorted({row['file_name'] for row in self.rows})}")
        print(f"Supplier memory refreshed: {self.memory_refresh_stats}")

    def selected_rows(self, file_filter: str | None) -> list[dict[str, Any]]:
        if not file_filter:
            return self.rows

        target = normalize_file_name(file_filter)

        return [
            row for row in self.rows
            if normalize_file_name(row["file_name"]) == target
        ]

    def compute_doc_fingerprint(self, rows: list[dict[str, Any]]) -> str:
        stable_rows = []

        for row in sorted(rows, key=lambda x: x["chunk_id"]):
            stable_rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "file_name": row["file_name"],
                    "page_number": row["page_number"],
                    "text": row["text"],
                    "extraction_method": row["extraction_method"],
                }
            )

        return sha256_text(json.dumps(stable_rows, sort_keys=True, ensure_ascii=False))

    def retrieve(self, question: str, top_k: int = 6, file_filter: str | None = None) -> list[dict[str, Any]]:
        candidates = self.selected_rows(file_filter)

        if not candidates:
            return []

        candidate_ids = {row["chunk_id"] for row in candidates}

        query_vec = self.embedder.encode(
            "query: " + question,
            normalize_embeddings=True,
        ).tolist()

        dense_limit = min(max(top_k * 20, 50), max(len(self.rows), 1))
        dense_df = self.table.search(query_vec).limit(dense_limit).to_pandas()

        dense_rank = {}

        for rank, (_, row) in enumerate(dense_df.iterrows(), start=1):
            chunk_id = str(row["chunk_id"])

            if chunk_id in candidate_ids:
                dense_rank[chunk_id] = rank

        bm25_scores = self.bm25.get_scores(tokenize(question))
        bm25_rank = {}

        row_by_id = {row["chunk_id"]: row for row in self.rows}

        sorted_bm25 = sorted(
            enumerate(bm25_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        bm25_position = 1

        for idx, _ in sorted_bm25:
            row = self.rows[idx]
            chunk_id = row["chunk_id"]

            if chunk_id not in candidate_ids:
                continue

            bm25_rank[chunk_id] = bm25_position
            bm25_position += 1

            if len(bm25_rank) >= max(top_k * 3, 10):
                break

        scores = {}

        for chunk_id, rank in dense_rank.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)

        for chunk_id, rank in bm25_rank.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)

        q_lower = question.lower()

        for row in candidates:
            chunk_id = row["chunk_id"]
            text_lower = row["text"].lower()

            if ("email" in q_lower or "contact" in q_lower) and ("email" in text_lower or "primary contact" in text_lower):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.08

            if ("payment" in q_lower or "terms" in q_lower) and ("payment terms" in text_lower or "net 30" in text_lower):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.08

            if ("food safety" in q_lower or "safety" in q_lower) and "food safety" in text_lower:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.10

            if ("policy number" in q_lower or "policy" in q_lower) and "policy number" in text_lower:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.08

            if ("bank name" in q_lower or "bank" in q_lower) and "bank name" in text_lower:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.08

            if ("who signed" in q_lower or "signed by" in q_lower) and "signed by" in text_lower:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.08

        if not scores:
            for row in candidates[:top_k]:
                scores[row["chunk_id"]] = 0.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []

        for chunk_id, score in ranked[:top_k]:
            row = dict(row_by_id[chunk_id])
            row["score"] = float(score)
            results.append(row)

        return results

    def choose_answer_mode(self, requested_mode: str, question: str, file_filter: str | None) -> str:
        mode = requested_mode.lower().strip()

        if mode in {"compliance", "rules"}:
            return "compliance"

        if mode in {"facts", "fact", "extractive"}:
            return "facts"

        if mode in {"rag", "general", "llm"}:
            return "rag"

        selected = self.selected_rows(file_filter)
        selected_files = {row["file_name"] for row in selected}

        if is_compliance_question(question):
            if file_filter or len(selected_files) <= 1:
                return "compliance"

        if is_fact_question(question):
            if file_filter or len(selected_files) <= 1:
                return "facts"

        return "rag"
runtime = WarmRuntime()
app = FastAPI(title="LocalDocLens Warm Hybrid RAG Server")

# Local-first runtime security:
# - localhost-only by default
# - optional X-LocalDoc-Token API token
# - restrictive CORS
# - basic security headers
install_cors(app)
install_runtime_security(app)



@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(runtime.rows),
        "files_indexed": sorted({row["file_name"] for row in runtime.rows}),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cache_version": CACHE_VERSION,
        "ollama_model": OLLAMA_MODEL,
    }


@app.get("/cache/stats")
def cache_stats():
    return runtime.cache.stats()


@app.get("/memory/stats")
def memory_stats():
    return runtime.memory.stats()



@app.get("/learning/stats")
def learning_stats():
    return runtime.learning.stats()


@app.post("/learning/export")
def learning_export():
    return runtime.learning.export_jsonl()


@app.post("/cache/verify")
def verify_cache(req: VerifyCacheRequest):
    ok = runtime.cache.verify(
        cache_key=req.cache_key,
        verified_by=req.verified_by,
        note=req.note,
    )

    return {
        "ok": ok,
        "cache_key": req.cache_key,
        "verified_by": req.verified_by,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()

    file_filter = req.file.strip() or None
    selected = runtime.selected_rows(file_filter)

    if selected:
        runtime.memory.refresh(selected)

    if not selected:
        total_latency = time.perf_counter() - start

        return AskResponse(
            question=req.question,
            answer=(
                f"No indexed evidence found for file '{req.file}'. "
                "Run localdoc inspect to see indexed files, or run localdoc ingest data/docs."
            ),
            latency_s=round(total_latency, 4),
            retrieval_latency_s=0.0,
            generation_latency_s=0.0,
            evidence=[],
            cache_status="file_not_found",
            cache_key="",
            cache_verified=False,
            answer_mode="none",
            file_filter=file_filter,
        )

    answer_mode = runtime.choose_answer_mode(req.mode, req.question, file_filter)
    doc_fingerprint = runtime.compute_doc_fingerprint(selected)

    cache_key = runtime.cache.make_key(
        doc_fingerprint=doc_fingerprint,
        question=req.question,
        top_k=req.top_k,
        file_filter=file_filter,
        answer_mode=answer_mode,
        use_llm=req.use_llm,
    )

    if req.use_cache:
        cached = runtime.cache.get(
            cache_key=cache_key,
            require_verified=req.require_verified,
        )

        if cached is not None:
            total_latency = time.perf_counter() - start

            return AskResponse(
                question=req.question,
                answer=cached["answer"],
                latency_s=round(total_latency, 4),
                retrieval_latency_s=0.0,
                generation_latency_s=0.0,
                evidence=cached["evidence"],
                cache_status="hit_verified" if cached["verified"] else "hit_unverified",
                cache_key=cache_key,
                cache_verified=cached["verified"],
                answer_mode=answer_mode,
                file_filter=file_filter,
            )

    if answer_mode == "compliance":
        generation_start = time.perf_counter()
        answer, evidence = make_compliance_answer(req.question, selected)
        generation_latency = time.perf_counter() - generation_start
        retrieval_latency = 0.0
    elif answer_mode == "facts":
        generation_start = time.perf_counter()
        answer, evidence = make_fact_answer(req.question, selected)
        generation_latency = time.perf_counter() - generation_start
        retrieval_latency = 0.0

        if answer is None:
            answer_mode = "rag"
            retrieval_start = time.perf_counter()
            evidence = runtime.retrieve(
                req.question,
                top_k=req.top_k,
                file_filter=file_filter,
            )
            retrieval_latency = time.perf_counter() - retrieval_start

            generation_start = time.perf_counter()
            answer = make_general_rag_answer(
                question=req.question,
                evidence=evidence,
                use_llm=req.use_llm,
            )
            generation_latency = time.perf_counter() - generation_start
    else:
        retrieval_start = time.perf_counter()
        evidence = runtime.retrieve(
            req.question,
            top_k=req.top_k,
            file_filter=file_filter,
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        generation_start = time.perf_counter()
        answer = make_general_rag_answer(
            question=req.question,
            evidence=evidence,
            use_llm=req.use_llm,
        )
        generation_latency = time.perf_counter() - generation_start

    verification = verify_answer(
        question=req.question,
        answer=answer,
        evidence=evidence,
        answer_mode=answer_mode,
        file_filter=file_filter,
        selected_rows=selected,
    )

    learning_example_id = runtime.learning.record_example(
        question=req.question,
        answer=answer,
        answer_mode=answer_mode,
        evidence=evidence,
        verification=verification,
    )

    runtime.cache.set(
        cache_key=cache_key,
        doc_fingerprint=doc_fingerprint,
        question=req.question,
        top_k=req.top_k,
        answer=answer,
        evidence=evidence,
    )

    total_latency = time.perf_counter() - start

    return AskResponse(
        question=req.question,
        answer=answer,
        latency_s=round(total_latency, 4),
        retrieval_latency_s=round(retrieval_latency, 4),
        generation_latency_s=round(generation_latency, 4),
        evidence=evidence,
        cache_status="miss_stored",
        cache_key=cache_key,
        cache_verified=False,
        answer_mode=answer_mode,
        file_filter=file_filter,
    )
