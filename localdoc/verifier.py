import hashlib
import re
from typing import Any


def compact(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_text(text: str) -> str:
    return compact(text).lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        compact(item.get("text", ""))
        for item in evidence
        if item.get("text")
    )


def first_source(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {}

    return evidence[0]


def clean_answer_value(answer: str) -> str:
    text = compact(answer)

    text = re.sub(r"^direct answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\bsource\s*:", text, flags=re.IGNORECASE)[0].strip()
    text = re.split(r"\bevidence\s*:", text, flags=re.IGNORECASE)[0].strip()

    if "\n" in text:
        text = text.split("\n", 1)[0].strip()

    text = text.strip(" .")

    return text


def detect_question_type(question: str) -> str:
    q = question.lower()

    if "email" in q:
        return "email"

    if "phone" in q:
        return "phone"

    if "payment terms" in q or "payment term" in q:
        return "payment_terms"

    if "bank name" in q:
        return "bank_name"

    if "routing" in q:
        return "routing_number"

    if "account" in q and "ending" in q:
        return "account_ending"

    if "policy number" in q:
        return "policy_number"

    if "policy carrier" in q or "insurance carrier" in q:
        return "policy_carrier"

    if "expiration" in q and ("insurance" in q or "policy" in q):
        return "expiration_date"

    if "who signed" in q or "signed by" in q or "signer" in q:
        return "signers"

    if "food safety" in q or "safety certificate" in q or "safety certification" in q:
        return "food_safety"

    if "w-9" in q or "w9" in q:
        return "w9"

    if "insurance" in q or "expired" in q or "coi" in q:
        return "insurance_status"

    if "signature" in q or "agreement" in q or "signed" in q:
        return "agreement_status"

    if "risk" in q or "approve" in q or "approval" in q or "decision" in q:
        return "risk_decision"

    return "general"


def extract_candidates_from_text(question_type: str, text: str) -> set[str]:
    candidates = set()

    if not text:
        return candidates

    if question_type == "email":
        for match in re.findall(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text):
            candidates.add(match.strip())

    elif question_type == "phone":
        for match in re.findall(r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}", text):
            candidates.add(compact(match))

    elif question_type == "payment_terms":
        for match in re.findall(r"\bNet\s*\d+\b", text, flags=re.IGNORECASE):
            candidates.add(compact(match))

    elif question_type == "routing_number":
        for match in re.findall(r"Routing number\s*[:\-]?\s*([0-9]{6,12})", text, flags=re.IGNORECASE):
            candidates.add(match.strip())

    elif question_type == "account_ending":
        for match in re.findall(
            r"(?:Account number\s*[:\-]?\s*ending in|account ending)\s*([0-9]{2,8})",
            text,
            flags=re.IGNORECASE,
        ):
            candidates.add(match.strip())

    elif question_type == "policy_number":
        for match in re.findall(r"Policy number\s*[:\-]?\s*([A-Z0-9\-]+)", text, flags=re.IGNORECASE):
            candidates.add(match.strip())

    elif question_type == "expiration_date":
        for match in re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text):
            candidates.add(match.strip())

    elif question_type == "bank_name":
        for match in re.findall(
            r"Bank name\s*[:\-]?\s*([^\n\r]+?)(?:\s+Routing number|\s+Account number|\s+ACH|\s+Payment terms|$)",
            text,
            flags=re.IGNORECASE,
        ):
            value = compact(match).strip(" .,:;-")
            if value:
                candidates.add(value)

    elif question_type == "policy_carrier":
        for match in re.findall(
            r"Policy carrier\s*[:\-]?\s*([^\n\r]+?)(?:\s+Policy number|\s+Coverage|\s+Expiration|$)",
            text,
            flags=re.IGNORECASE,
        ):
            value = compact(match).strip(" .,:;-")
            if value:
                candidates.add(value)

    elif question_type == "signers":
        for match in re.findall(r"Signed by\s+([A-Z][A-Za-z .'-]+?)\s+on\s+\d{2}/\d{2}/\d{4}", text):
            value = compact(match).strip(" .,:;-")
            if value:
                candidates.add(value)

    return candidates


def answer_contains_candidate(answer: str, candidates: set[str]) -> bool:
    answer_norm = normalize_text(answer)

    for candidate in candidates:
        if normalize_text(candidate) in answer_norm:
            return True

    return False


def answer_exactly_supported(answer: str, evidence: str) -> bool:
    answer_value = clean_answer_value(answer)

    if not answer_value:
        return False

    answer_norm = normalize_text(answer_value)
    evidence_norm = normalize_text(evidence)

    if answer_norm and answer_norm in evidence_norm:
        return True

    pieces = re.split(r"[,;]| and ", answer_value)
    useful_pieces = []

    for piece in pieces:
        piece = compact(piece)
        piece = re.sub(
            r"^(buyer signer|supplier signer)\s*:\s*",
            "",
            piece,
            flags=re.IGNORECASE,
        ).strip()

        if len(piece) >= 3:
            useful_pieces.append(piece)

    if useful_pieces and all(normalize_text(piece) in evidence_norm for piece in useful_pieces):
        return True

    return False


def strong_yes_no_support(question_type: str, answer: str, evidence: str) -> bool:
    answer_lower = answer.lower()
    evidence_lower = evidence.lower()

    if question_type == "food_safety":
        if "yes" in answer_lower and (
            "food safety certificate: received" in evidence_lower
            or "food safety certificate received" in evidence_lower
            or "certificate review status: passed" in evidence_lower
        ):
            return True

    if question_type == "insurance_status":
        if ("not appear" in answer_lower or "not expired" in answer_lower or "active" in answer_lower) and (
            "not expired" in evidence_lower
            or "active" in evidence_lower
            or "valid" in evidence_lower
        ):
            return True

        if ("expired" in answer_lower or "insurance issue" in answer_lower) and (
            "appears expired" in evidence_lower
            or "is expired" in evidence_lower
            or "coverage expired" in evidence_lower
        ):
            return True

    if question_type == "w9":
        if ("missing" in answer_lower or "not received" in answer_lower) and (
            "w-9 missing" in evidence_lower
            or "w-9 tax form missing" in evidence_lower
            or "not received" in evidence_lower
        ):
            return True

        if ("received" in answer_lower or "appears to have" in answer_lower) and (
            "w-9 received" in evidence_lower
            or "signed w-9 received" in evidence_lower
            or "w-9 tax form received" in evidence_lower
        ):
            return True

    if question_type == "agreement_status":
        if ("missing" in answer_lower or "unsigned" in answer_lower) and (
            "signature missing" in evidence_lower
            or "missing supplier signature" in evidence_lower
            or "pending signature" in evidence_lower
        ):
            return True

        if ("signed" in answer_lower or "complete" in answer_lower) and (
            "agreement status: signed" in evidence_lower
            or "supplier representative signature signed" in evidence_lower
            or "signed by both parties" in evidence_lower
        ):
            return True

    return False


def high_quality_source(evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return False

    source = first_source(evidence)
    extraction_method = str(source.get("extraction_method", "")).lower()
    ocr_confidence = source.get("ocr_confidence")

    if extraction_method == "pdf_text":
        return True

    try:
        if ocr_confidence is not None and float(ocr_confidence) >= 0.85:
            return True
    except Exception:
        pass

    return False


def detect_contradiction(
    question_type: str,
    answer: str,
    selected_rows: list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    if not selected_rows:
        return False, "No full-document contradiction scan available."

    if question_type not in {
        "email",
        "phone",
        "payment_terms",
        "bank_name",
        "routing_number",
        "account_ending",
        "policy_number",
        "policy_carrier",
        "expiration_date",
    }:
        return False, "Question type does not require strict contradiction scan."

    full_text = "\n\n".join(row.get("text", "") for row in selected_rows)
    candidates = extract_candidates_from_text(question_type, full_text)

    if len(candidates) <= 1:
        return False, "No conflicting candidate values found."

    if answer_contains_candidate(answer, candidates):
        answer_norm = normalize_text(answer)
        matching = {
            candidate
            for candidate in candidates
            if normalize_text(candidate) in answer_norm
        }

        non_matching = candidates - matching

        if non_matching:
            return True, f"Multiple candidate values found: {sorted(candidates)}"

    return True, f"Multiple candidate values found: {sorted(candidates)}"


def verify_answer(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    answer_mode: str,
    file_filter: str | None = None,
    selected_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    question_type = detect_question_type(question)
    ev_text = evidence_text(evidence)
    source = first_source(evidence)

    has_evidence = bool(ev_text.strip())
    has_source_page = bool(source.get("file_name")) and source.get("page_number") is not None
    exact_supported = answer_exactly_supported(answer, ev_text)
    candidates = extract_candidates_from_text(question_type, ev_text)
    candidate_supported = answer_contains_candidate(answer, candidates)
    yes_no_supported = strong_yes_no_support(question_type, answer, ev_text)
    quality_source = high_quality_source(evidence)

    contradiction, contradiction_reason = detect_contradiction(
        question_type=question_type,
        answer=answer,
        selected_rows=selected_rows,
    )

    support_found = exact_supported or candidate_supported or yes_no_supported

    confidence = 0.0

    if has_evidence:
        confidence += 0.20

    if has_source_page:
        confidence += 0.20

    if exact_supported:
        confidence += 0.35
    elif candidate_supported:
        confidence += 0.30
    elif yes_no_supported:
        confidence += 0.25

    if quality_source:
        confidence += 0.10

    if answer_mode in {"facts", "compliance"}:
        confidence += 0.10

    if question_type != "general":
        confidence += 0.05

    if contradiction:
        confidence -= 0.35

    confidence = max(0.0, min(1.0, confidence))

    if contradiction:
        status = "contradicted"
        reason = contradiction_reason
    elif support_found and has_source_page and confidence >= 0.85:
        status = "auto_verified"
        reason = "Answer is directly supported by source evidence with page citation."
    elif support_found and has_source_page:
        status = "needs_review"
        reason = "Answer appears supported by evidence, but confidence is not high enough for automatic verification."
    elif has_evidence and has_source_page:
        status = "needs_review"
        reason = "Evidence and source page exist, but the answer was not directly matched to the evidence."
    else:
        status = "unverified"
        reason = "Missing evidence or source page."

    return {
        "status": status,
        "confidence": round(confidence, 4),
        "reason": reason,
        "question_type": question_type,
        "answer_hash": sha256_text(answer),
        "evidence_hash": sha256_text(ev_text),
        "file_name": source.get("file_name"),
        "page_number": source.get("page_number"),
        "checks": {
            "has_evidence": has_evidence,
            "has_source_page": has_source_page,
            "exact_supported": exact_supported,
            "candidate_supported": candidate_supported,
            "yes_no_supported": yes_no_supported,
            "quality_source": quality_source,
            "contradiction": contradiction,
            "contradiction_reason": contradiction_reason,
            "answer_mode": answer_mode,
            "file_filter": file_filter,
        },
    }
