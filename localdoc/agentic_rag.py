import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from localdoc.batch_analyze import clean_text, filter_rows, load_rows, sha256_text
from localdoc.facts import extract_facts_for_file


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def tokenize(value: str) -> list[str]:
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9@._/-]+", " ", value)
    return [token for token in value.split() if len(token) >= 2]


def safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def compact_evidence_text(text: str, max_chars: int = 900) -> str:
    return clean_text(text, max_chars=max_chars)


def get_nested_value(obj: Any, path: list[str]) -> Any:
    current = obj

    for part in path:
        if not isinstance(current, dict):
            return None

        if part not in current:
            return None

        current = current[part]

    if isinstance(current, dict) and "value" in current:
        return current.get("value")

    return current


def extract_fact_source(obj: Any, path: list[str]) -> dict[str, Any]:
    current = obj

    for part in path:
        if not isinstance(current, dict):
            return {}

        if part not in current:
            return {}

        current = current[part]

    if isinstance(current, dict):
        return current.get("source") or {}

    return {}


def collect_fact_sources(obj: Any) -> list[dict[str, Any]]:
    sources = []

    def walk(value: Any):
        if isinstance(value, dict):
            if value.get("source") and value.get("evidence_text"):
                sources.append(value)

            for nested in value.values():
                walk(nested)

        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(obj)
    return sources


def infer_question_intent(question: str) -> dict[str, Any]:
    q = normalize_text(question)

    if any(word in q for word in ["approve", "approval", "reject", "onboard", "risk", "compliance status", "compliant"]):
        return {
            "intent": "supplier_approval_decision",
            "description": "Determine whether supplier should be approved using compliance evidence.",
        }

    if "bank" in q:
        return {
            "intent": "bank_verification",
            "description": "Answer a bank verification or bank detail question.",
        }

    if "payment terms" in q or "net 30" in q or "invoice" in q:
        return {
            "intent": "payment_terms",
            "description": "Answer a payment terms question.",
        }

    if "insurance" in q or "policy" in q or "coi" in q:
        return {
            "intent": "insurance_status",
            "description": "Answer an insurance status or policy question.",
        }

    if "w-9" in q or "w9" in q or "tax" in q:
        return {
            "intent": "w9_status",
            "description": "Answer a W-9 or tax form question.",
        }

    if "signature" in q or "agreement" in q or "signed" in q:
        return {
            "intent": "agreement_signature",
            "description": "Answer a supplier agreement/signature question.",
        }

    if "email" in q or "contact" in q or "phone" in q:
        return {
            "intent": "supplier_contact",
            "description": "Answer a supplier contact information question.",
        }

    if "food safety" in q or "sanctions" in q or "conflict of interest" in q:
        return {
            "intent": "supplemental_compliance",
            "description": "Answer supplemental compliance question.",
        }

    if "missing" in q or "expired" in q:
        return {
            "intent": "exception_search",
            "description": "Find missing, expired, or exception items.",
        }

    return {
        "intent": "general_supplier_question",
        "description": "General supplier-document question requiring retrieval and answer verification.",
    }


def make_agent_plan(question: str, intent: dict[str, Any]) -> list[dict[str, Any]]:
    intent_name = intent["intent"]

    plan = [
        {
            "step_id": "plan_001",
            "agent": "planner",
            "tool": "classify_intent",
            "objective": "Classify the user's question and decide which evidence checks are required.",
            "query": question,
        },
        {
            "step_id": "memory_001",
            "agent": "memory_agent",
            "tool": "extract_supplier_memory",
            "objective": "Load structured supplier facts from the indexed document.",
            "query": question,
        },
    ]

    if intent_name == "supplier_approval_decision":
        plan.extend(
            [
                {
                    "step_id": "retrieve_w9",
                    "agent": "retriever_agent",
                    "tool": "retrieve_evidence",
                    "objective": "Find W-9 evidence.",
                    "query": "W-9 tax form received missing signed taxpayer identification",
                    "required_terms": ["w-9"],
                },
                {
                    "step_id": "retrieve_insurance",
                    "agent": "retriever_agent",
                    "tool": "retrieve_evidence",
                    "objective": "Find certificate of insurance evidence.",
                    "query": "certificate of insurance policy expiration expired active coverage",
                    "required_terms": ["insurance", "policy"],
                },
                {
                    "step_id": "retrieve_bank",
                    "agent": "retriever_agent",
                    "tool": "retrieve_evidence",
                    "objective": "Find bank verification evidence.",
                    "query": "bank verification ACH routing payment terms verified",
                    "required_terms": ["bank", "verification"],
                },
                {
                    "step_id": "retrieve_agreement",
                    "agent": "retriever_agent",
                    "tool": "retrieve_evidence",
                    "objective": "Find supplier agreement signature evidence.",
                    "query": "supplier agreement signature signed missing pending",
                    "required_terms": ["agreement", "signature"],
                },
                {
                    "step_id": "retrieve_supplemental",
                    "agent": "retriever_agent",
                    "tool": "retrieve_evidence",
                    "objective": "Find sanctions, food safety, and conflict-of-interest evidence.",
                    "query": "food safety sanctions screening conflict of interest compliance passed clear",
                    "required_terms": ["sanctions", "conflict", "food"],
                },
            ]
        )

    elif intent_name == "bank_verification":
        plan.append(
            {
                "step_id": "retrieve_bank",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find bank verification and ACH evidence.",
                "query": "bank name bank verification ACH routing payment terms verified",
                "required_terms": ["bank"],
            }
        )

    elif intent_name == "payment_terms":
        plan.append(
            {
                "step_id": "retrieve_payment_terms",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find payment terms evidence.",
                "query": "payment terms net invoice due date remittance",
                "required_terms": ["payment", "terms"],
            }
        )

    elif intent_name == "insurance_status":
        plan.append(
            {
                "step_id": "retrieve_insurance",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find insurance status, policy number, and expiration evidence.",
                "query": "certificate of insurance policy number expiration active expired coverage",
                "required_terms": ["insurance", "policy"],
            }
        )

    elif intent_name == "w9_status":
        plan.append(
            {
                "step_id": "retrieve_w9",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find W-9 evidence.",
                "query": "W-9 tax form signed received missing taxpayer identification",
                "required_terms": ["w-9"],
            }
        )

    elif intent_name == "agreement_signature":
        plan.append(
            {
                "step_id": "retrieve_agreement",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find supplier agreement signature evidence.",
                "query": "supplier agreement signature signed by missing pending",
                "required_terms": ["agreement", "signature"],
            }
        )

    elif intent_name == "supplier_contact":
        plan.append(
            {
                "step_id": "retrieve_contact",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find supplier contact evidence.",
                "query": "supplier legal name primary contact email phone",
                "required_terms": ["email", "contact"],
            }
        )

    elif intent_name == "supplemental_compliance":
        plan.append(
            {
                "step_id": "retrieve_supplemental",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find supplemental compliance evidence.",
                "query": "food safety sanctions screening conflict of interest passed clear review",
                "required_terms": ["food", "sanctions", "conflict"],
            }
        )

    elif intent_name == "exception_search":
        plan.append(
            {
                "step_id": "retrieve_exceptions",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Find missing, expired, failed, or exception evidence.",
                "query": "missing expired failed not received pending signature exception",
                "required_terms": ["missing", "expired", "failed"],
            }
        )

    else:
        plan.append(
            {
                "step_id": "retrieve_general",
                "agent": "retriever_agent",
                "tool": "retrieve_evidence",
                "objective": "Retrieve evidence most relevant to the user's question.",
                "query": question,
                "required_terms": [],
            }
        )

    plan.extend(
        [
            {
                "step_id": "reason_001",
                "agent": "reasoner_agent",
                "tool": "reason_over_evidence",
                "objective": "Use memory and retrieved evidence to create a grounded answer.",
                "query": question,
            },
            {
                "step_id": "verify_001",
                "agent": "verifier_agent",
                "tool": "verify_answer",
                "objective": "Check whether the final answer is supported by source evidence.",
                "query": question,
            },
        ]
    )

    return plan


def score_row(row: dict[str, Any], query: str, required_terms: list[str] | None = None) -> float:
    text = normalize_text(row.get("text", ""))
    q = normalize_text(query)

    query_tokens = tokenize(q)
    text_tokens = set(tokenize(text))

    if not query_tokens:
        return 0.0

    score = 0.0

    for token in query_tokens:
        if token in text_tokens:
            score += 2.0

    for phrase in re.findall(r"[a-z0-9@._/-]+(?:\s+[a-z0-9@._/-]+)+", q):
        if len(phrase) >= 6 and phrase in text:
            score += 5.0

    if required_terms:
        for term in required_terms:
            term_norm = normalize_text(term)
            if term_norm and term_norm in text:
                score += 4.0

    # Procurement-specific boosts.
    boosts = {
        "w-9": ["w-9", "w9", "tax form", "taxpayer"],
        "insurance": ["insurance", "policy", "coverage", "expiration"],
        "bank": ["bank", "ach", "routing", "payment terms"],
        "agreement": ["agreement", "signature", "signed"],
        "sanctions": ["sanctions", "screening"],
        "food": ["food safety", "certificate"],
        "contact": ["primary contact", "email", "phone"],
    }

    for concept, phrases in boosts.items():
        if concept in q:
            for phrase in phrases:
                if phrase in text:
                    score += 2.0

    return score


def retrieve_evidence(
    rows: list[dict[str, Any]],
    query: str,
    required_terms: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    scored = []

    for row in rows:
        score = score_row(row, query=query, required_terms=required_terms)

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda item: (-item[0], item[1]["file_name"], item[1]["page_number"]))

    evidence = []

    for score, row in scored[:top_k]:
        evidence.append(
            {
                "file_name": row["file_name"],
                "page_number": row["page_number"],
                "chunk_id": row.get("chunk_id"),
                "score": round(score, 4),
                "extraction_method": row.get("extraction_method"),
                "ocr_confidence": row.get("ocr_confidence"),
                "text": compact_evidence_text(row.get("text", ""), max_chars=1200),
            }
        )

    return evidence


def facts_brief(facts: dict[str, Any]) -> dict[str, Any]:
    risk = facts.get("risk", {}) if isinstance(facts, dict) else {}

    brief = {
        "supplier_name": get_nested_value(facts, ["supplier_name"]),
        "overall_risk": risk.get("overall_risk"),
        "decision": risk.get("decision"),
        "failed_items": risk.get("failed_items"),
        "review_items": risk.get("review_items"),
        "w9_status": get_nested_value(facts, ["w9_status"]),
        "insurance_status": get_nested_value(facts, ["insurance", "status"]),
        "insurance_expiration": get_nested_value(facts, ["insurance", "expiration_date"]),
        "insurance_policy_number": get_nested_value(facts, ["insurance", "policy_number"]),
        "bank_name": get_nested_value(facts, ["bank", "bank_name"]),
        "bank_verification_status": get_nested_value(facts, ["bank", "verification_status"]),
        "payment_terms": get_nested_value(facts, ["bank", "payment_terms"]),
        "agreement_status": get_nested_value(facts, ["agreement", "status"]),
        "agreement_signers": get_nested_value(facts, ["agreement", "signers"]),
        "primary_email": get_nested_value(facts, ["contact", "email"]),
        "food_safety_status": get_nested_value(facts, ["supplemental", "food_safety_status"]),
        "sanctions_status": get_nested_value(facts, ["supplemental", "sanctions_status"]),
        "conflict_of_interest_status": get_nested_value(facts, ["supplemental", "conflict_of_interest_status"]),
    }

    return {key: value for key, value in brief.items() if value not in [None, "", [], {}]}


def flatten_evidence(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []

    seen = set()

    for result in tool_results:
        if result.get("tool") != "retrieve_evidence":
            continue

        for item in result.get("output", []):
            key = (item.get("file_name"), item.get("page_number"), item.get("chunk_id"))

            if key in seen:
                continue

            seen.add(key)
            evidence.append(item)

    return evidence


def first_evidence_for_keywords(evidence: list[dict[str, Any]], keywords: list[str]) -> dict[str, Any] | None:
    best = None
    best_score = -1

    for item in evidence:
        text = normalize_text(item.get("text", ""))
        score = 0

        for keyword in keywords:
            k = normalize_text(keyword)

            if k and k in text:
                score += 5

            for token in tokenize(k):
                if token in text:
                    score += 1

        if score > best_score:
            best = item
            best_score = score

    if best_score <= 0:
        return evidence[0] if evidence else None

    return best


def compose_deterministic_answer(
    question: str,
    intent: dict[str, Any],
    facts: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    intent_name = intent["intent"]
    brief = facts_brief(facts)
    risk = facts.get("risk", {}) if isinstance(facts, dict) else {}

    sources = []

    def add_source(item: dict[str, Any] | None):
        if not item:
            return

        source = {
            "file_name": item.get("file_name"),
            "page_number": item.get("page_number"),
            "text": item.get("text"),
        }

        key = (source["file_name"], source["page_number"])

        if key not in [(s["file_name"], s["page_number"]) for s in sources]:
            sources.append(source)

    if intent_name == "supplier_approval_decision":
        supplier_name = brief.get("supplier_name") or "The supplier"
        overall_risk = brief.get("overall_risk") or "Unknown"
        decision = brief.get("decision") or ""

        failed_items = risk.get("failed_items", []) or []
        review_items = risk.get("review_items", []) or []

        answer_lines = [
            f"{supplier_name} should be treated as {overall_risk} risk.",
        ]

        if decision:
            answer_lines.append(f"Decision: {decision}")

        if failed_items:
            answer_lines.append("Failed items: " + ", ".join(str(item) for item in failed_items) + ".")

        if review_items:
            answer_lines.append("Review items: " + ", ".join(str(item) for item in review_items) + ".")

        checks = []

        if "w9_status" in brief:
            checks.append(f"W-9: {brief['w9_status']}")

        if "insurance_status" in brief:
            detail = str(brief["insurance_status"])

            if brief.get("insurance_expiration"):
                detail += f" through {brief['insurance_expiration']}"

            checks.append(f"Insurance: {detail}")

        if "bank_verification_status" in brief:
            checks.append(f"Bank verification: {brief['bank_verification_status']}")

        if "agreement_status" in brief:
            checks.append(f"Agreement: {brief['agreement_status']}")

        if checks:
            answer_lines.append("Evidence checks: " + "; ".join(checks) + ".")

        for keywords in [
            ["w-9", "tax form"],
            ["insurance", "policy", "expiration"],
            ["bank", "verification"],
            ["agreement", "signature"],
            ["sanctions", "food safety", "conflict"],
        ]:
            add_source(first_evidence_for_keywords(evidence, keywords))

        return {
            "answer": "\n".join(answer_lines),
            "sources": sources,
            "answer_mode": "agentic_compliance_decision",
        }

    if intent_name == "bank_verification":
        bank_name = brief.get("bank_name")
        status = brief.get("bank_verification_status")
        payment_terms = brief.get("payment_terms")

        add_source(first_evidence_for_keywords(evidence, ["bank name", "bank verification", "payment terms"]))

        parts = []

        if bank_name:
            parts.append(f"Bank name: {bank_name}.")

        if status:
            parts.append(f"Bank verification status: {status}.")

        if payment_terms:
            parts.append(f"Payment terms: {payment_terms}.")

        return {
            "answer": " ".join(parts) if parts else "I found bank-related evidence, but could not confidently extract the exact bank answer.",
            "sources": sources,
            "answer_mode": "agentic_factual_bank",
        }

    if intent_name == "payment_terms":
        payment_terms = brief.get("payment_terms")
        add_source(first_evidence_for_keywords(evidence, ["payment terms", "invoice", "net"]))

        return {
            "answer": f"Payment terms: {payment_terms}." if payment_terms else "I could not confidently extract payment terms from the evidence.",
            "sources": sources,
            "answer_mode": "agentic_factual_payment_terms",
        }

    if intent_name == "insurance_status":
        status = brief.get("insurance_status")
        expiration = brief.get("insurance_expiration")
        policy = brief.get("insurance_policy_number")

        add_source(first_evidence_for_keywords(evidence, ["insurance", "policy", "expiration"]))

        parts = []

        if status:
            parts.append(f"Insurance status: {status}.")

        if expiration:
            parts.append(f"Expiration date: {expiration}.")

        if policy:
            parts.append(f"Policy number: {policy}.")

        return {
            "answer": " ".join(parts) if parts else "I could not confidently extract the insurance status from the evidence.",
            "sources": sources,
            "answer_mode": "agentic_factual_insurance",
        }

    if intent_name == "w9_status":
        status = brief.get("w9_status")
        add_source(first_evidence_for_keywords(evidence, ["w-9", "tax form", "taxpayer"]))

        return {
            "answer": f"W-9 status: {status}." if status else "I could not confidently extract the W-9 status from the evidence.",
            "sources": sources,
            "answer_mode": "agentic_factual_w9",
        }

    if intent_name == "agreement_signature":
        status = brief.get("agreement_status")
        signers = brief.get("agreement_signers")
        add_source(first_evidence_for_keywords(evidence, ["agreement", "signature", "signed"]))

        parts = []

        if status:
            parts.append(f"Agreement status: {status}.")

        if signers:
            if isinstance(signers, list):
                parts.append("Signers: " + ", ".join(str(item) for item in signers) + ".")
            else:
                parts.append(f"Signers: {signers}.")

        return {
            "answer": " ".join(parts) if parts else "I could not confidently extract the supplier agreement signature status from the evidence.",
            "sources": sources,
            "answer_mode": "agentic_factual_agreement",
        }

    if intent_name == "supplier_contact":
        email = brief.get("primary_email")
        supplier_name = brief.get("supplier_name")
        add_source(first_evidence_for_keywords(evidence, ["primary contact", "email", "phone"]))

        parts = []

        if supplier_name:
            parts.append(f"Supplier: {supplier_name}.")

        if email:
            parts.append(f"Email: {email}.")

        return {
            "answer": " ".join(parts) if parts else "I could not confidently extract contact information from the evidence.",
            "sources": sources,
            "answer_mode": "agentic_factual_contact",
        }

    if intent_name == "supplemental_compliance":
        add_source(first_evidence_for_keywords(evidence, ["food safety", "sanctions", "conflict of interest"]))

        parts = []

        for label, key in [
            ("Food safety", "food_safety_status"),
            ("Sanctions", "sanctions_status"),
            ("Conflict of interest", "conflict_of_interest_status"),
        ]:
            if brief.get(key):
                parts.append(f"{label}: {brief[key]}.")

        return {
            "answer": " ".join(parts) if parts else "I found supplemental compliance evidence, but could not confidently extract all statuses.",
            "sources": sources,
            "answer_mode": "agentic_factual_supplemental",
        }

    if intent_name == "exception_search":
        failed_items = risk.get("failed_items", []) or []
        review_items = risk.get("review_items", []) or []

        for keywords in [["missing"], ["expired"], ["failed"], ["pending"]]:
            add_source(first_evidence_for_keywords(evidence, keywords))

        parts = []

        if failed_items:
            parts.append("Failed items: " + ", ".join(str(item) for item in failed_items) + ".")

        if review_items:
            parts.append("Review items: " + ", ".join(str(item) for item in review_items) + ".")

        return {
            "answer": " ".join(parts) if parts else "I did not find clear failed or review items in the structured facts.",
            "sources": sources,
            "answer_mode": "agentic_exception_search",
        }

    # General fallback.
    for item in evidence[:3]:
        add_source(item)

    if evidence:
        answer = (
            "I found relevant source evidence, but this question does not match a specific structured compliance field. "
            "The most relevant evidence is from "
            + "; ".join(f"{src['file_name']} page {src['page_number']}" for src in sources)
            + "."
        )
    else:
        answer = "I could not find enough evidence to answer this question."

    return {
        "answer": answer,
        "sources": sources,
        "answer_mode": "agentic_general",
    }


def call_local_qwen(prompt: str, model: str = "qwen3:4b", timeout: int = 120) -> str | None:
    try:
        from localdoc.batch_analyze import call_ollama

        return call_ollama(prompt=prompt, timeout=timeout)

    except Exception:
        return None


def compose_with_llm(
    question: str,
    intent: dict[str, Any],
    facts: dict[str, Any],
    evidence: list[dict[str, Any]],
    deterministic_answer: dict[str, Any],
) -> dict[str, Any]:
    evidence_brief = []

    for item in evidence[:8]:
        evidence_brief.append(
            {
                "file_name": item.get("file_name"),
                "page_number": item.get("page_number"),
                "text": item.get("text"),
            }
        )

    prompt = f"""
You are LocalDocLens, a local-first procurement document agent.

Answer the user using ONLY the facts and evidence below.
Do not invent missing information.
Give a direct answer.
Mention source file/page references.
If evidence is weak, say that.

Question:
{question}

Intent:
{json.dumps(intent, indent=2)}

Structured facts:
{json.dumps(facts_brief(facts), indent=2)}

Retrieved evidence:
{json.dumps(evidence_brief, indent=2)}

Deterministic draft:
{deterministic_answer["answer"]}

Return the final answer only.
""".strip()

    llm_answer = call_local_qwen(prompt)

    if not llm_answer:
        return deterministic_answer

    result = dict(deterministic_answer)
    result["answer"] = llm_answer
    result["answer_mode"] = deterministic_answer.get("answer_mode", "agentic") + "_llm_composed"

    return result


def verify_answer_locally(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    facts: dict[str, Any],
) -> dict[str, Any]:
    answer_norm = normalize_text(answer)
    evidence_text = " ".join(str(source.get("text", "")) for source in sources)
    evidence_norm = normalize_text(evidence_text)
    brief = facts_brief(facts)

    has_sources = bool(sources)
    has_page = any(source.get("page_number") for source in sources)

    supported_hits = 0
    checks = {}

    for key, value in brief.items():
        if value in [None, "", [], {}]:
            continue

        if isinstance(value, list):
            candidates = [str(item) for item in value]
        else:
            candidates = [str(value)]

        for candidate in candidates:
            candidate_norm = normalize_text(candidate)

            if not candidate_norm or len(candidate_norm) < 2:
                continue

            if candidate_norm in answer_norm and candidate_norm in evidence_norm:
                supported_hits += 1
                checks[key] = "answer_and_evidence"
                break

            if candidate_norm in evidence_norm:
                checks[key] = "evidence_only"

    risk = facts.get("risk", {}) if isinstance(facts, dict) else {}
    failed_items = risk.get("failed_items", []) or []

    if failed_items:
        for item in failed_items:
            item_norm = normalize_text(str(item))

            if item_norm and item_norm in answer_norm:
                supported_hits += 1

    direct_quote_support = False

    answer_tokens = [token for token in tokenize(answer) if len(token) >= 4]
    evidence_tokens = set(tokenize(evidence_text))

    if answer_tokens:
        overlap = len([token for token in answer_tokens if token in evidence_tokens])
        overlap_ratio = overlap / max(1, len(answer_tokens))
        direct_quote_support = overlap_ratio >= 0.35
    else:
        overlap_ratio = 0.0

    confidence = 0.0

    if has_sources:
        confidence += 0.25

    if has_page:
        confidence += 0.20

    if supported_hits > 0:
        confidence += min(0.35, 0.12 * supported_hits)

    if direct_quote_support:
        confidence += 0.20

    confidence = round(min(1.0, confidence), 4)

    if confidence >= 0.75:
        status = "auto_verified"
        reason = "Answer is supported by source pages and extracted facts."
    elif confidence >= 0.45:
        status = "needs_review"
        reason = "Answer has some supporting evidence but should be reviewed."
    else:
        status = "unverified"
        reason = "Answer does not have enough evidence support."

    return {
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "checks": {
            "has_sources": has_sources,
            "has_page": has_page,
            "supported_fact_hits": supported_hits,
            "direct_quote_support": direct_quote_support,
            "token_overlap_ratio": round(overlap_ratio, 4),
            "fact_checks": checks,
        },
    }


def try_record_learning_example(
    question: str,
    answer: str,
    answer_mode: str,
    sources: list[dict[str, Any]],
    verification: dict[str, Any],
    intent: dict[str, Any],
):
    if verification.get("status") != "auto_verified":
        return None

    try:
        from localdoc.learning import LearningExampleStore

        source = sources[0] if sources else {}

        store = LearningExampleStore()

        return store.record_example(
            question=question,
            answer=answer,
            answer_mode=answer_mode,
            file_name=source.get("file_name"),
            page_number=source.get("page_number"),
            evidence_text=source.get("text", ""),
            verification_status=verification.get("status"),
            verification_confidence=verification.get("confidence"),
            verification_reason=verification.get("reason"),
            question_type=intent.get("intent"),
            checks=verification.get("checks", {}),
        )

    except Exception:
        return None


def run_agentic_rag(
    question: str,
    file_name: str | None = None,
    top_k: int = 5,
    use_llm: bool = False,
    max_retries: int = 1,
    output_dir: str = "artifacts/agent_runs",
) -> dict[str, Any]:
    start = time.perf_counter()

    question = question.strip()

    if not question:
        raise RuntimeError("Question cannot be empty.")

    rows = load_rows()
    selected_rows = filter_rows(rows, file_name)

    selected_file_name = file_name

    if selected_file_name is None:
        files = sorted({row["file_name"] for row in selected_rows})

        if len(files) == 1:
            selected_file_name = files[0]

    intent = infer_question_intent(question)
    plan = make_agent_plan(question, intent)

    trace = {
        "run_id": sha256_text(json.dumps({"question": question, "file": file_name, "at": utc_now()}, sort_keys=True)),
        "created_at": utc_now(),
        "question": question,
        "file_filter": file_name,
        "intent": intent,
        "plan": plan,
        "tool_results": [],
        "retries": [],
    }

    facts = {}

    for step in plan:
        tool = step["tool"]

        if tool == "classify_intent":
            trace["tool_results"].append(
                {
                    "step_id": step["step_id"],
                    "agent": step["agent"],
                    "tool": tool,
                    "objective": step["objective"],
                    "output": intent,
                }
            )

        elif tool == "extract_supplier_memory":
            if selected_file_name:
                facts = extract_facts_for_file(selected_file_name, selected_rows)
            else:
                # Multi-file fallback: use first file grouped result.
                first_file = sorted({row["file_name"] for row in selected_rows})[0]
                file_rows = [row for row in selected_rows if row["file_name"] == first_file]
                facts = extract_facts_for_file(first_file, file_rows)

            trace["tool_results"].append(
                {
                    "step_id": step["step_id"],
                    "agent": step["agent"],
                    "tool": tool,
                    "objective": step["objective"],
                    "output": facts_brief(facts),
                }
            )

        elif tool == "retrieve_evidence":
            evidence = retrieve_evidence(
                selected_rows,
                query=step.get("query", question),
                required_terms=step.get("required_terms", []),
                top_k=top_k,
            )

            trace["tool_results"].append(
                {
                    "step_id": step["step_id"],
                    "agent": step["agent"],
                    "tool": tool,
                    "objective": step["objective"],
                    "query": step.get("query"),
                    "output": evidence,
                }
            )

    evidence = flatten_evidence(trace["tool_results"])

    composed = compose_deterministic_answer(
        question=question,
        intent=intent,
        facts=facts,
        evidence=evidence,
    )

    if use_llm:
        composed = compose_with_llm(
            question=question,
            intent=intent,
            facts=facts,
            evidence=evidence,
            deterministic_answer=composed,
        )

    verification = verify_answer_locally(
        question=question,
        answer=composed["answer"],
        sources=composed.get("sources", []),
        facts=facts,
    )

    retry_count = 0

    while verification["status"] != "auto_verified" and retry_count < max_retries:
        retry_count += 1

        expanded_query = (
            question
            + " "
            + intent["intent"].replace("_", " ")
            + " "
            + " ".join(str(item) for item in facts_brief(facts).values())
        )

        retry_evidence = retrieve_evidence(
            selected_rows,
            query=expanded_query,
            required_terms=[],
            top_k=top_k + 3,
        )

        merged = evidence + retry_evidence

        seen = set()
        evidence = []

        for item in merged:
            key = (item.get("file_name"), item.get("page_number"), item.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)

        retry_composed = compose_deterministic_answer(
            question=question,
            intent=intent,
            facts=facts,
            evidence=evidence,
        )

        if use_llm:
            retry_composed = compose_with_llm(
                question=question,
                intent=intent,
                facts=facts,
                evidence=evidence,
                deterministic_answer=retry_composed,
            )

        retry_verification = verify_answer_locally(
            question=question,
            answer=retry_composed["answer"],
            sources=retry_composed.get("sources", []),
            facts=facts,
        )

        trace["retries"].append(
            {
                "retry": retry_count,
                "expanded_query": expanded_query,
                "previous_verification": verification,
                "new_verification": retry_verification,
                "new_evidence_count": len(retry_evidence),
            }
        )

        composed = retry_composed
        verification = retry_verification

    learning_example_id = try_record_learning_example(
        question=question,
        answer=composed["answer"],
        answer_mode=composed.get("answer_mode", "agentic_rag"),
        sources=composed.get("sources", []),
        verification=verification,
        intent=intent,
    )

    duration_s = round(time.perf_counter() - start, 4)

    trace["final"] = {
        "answer": composed["answer"],
        "answer_mode": composed.get("answer_mode"),
        "sources": composed.get("sources", []),
        "verification": verification,
        "learning_example_id": learning_example_id,
        "duration_s": duration_s,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trace_path = output_path / f"agent_trace_{trace['run_id'][:16]}.json"
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "question": question,
        "file_filter": file_name,
        "intent": intent,
        "answer": composed["answer"],
        "answer_mode": composed.get("answer_mode"),
        "verification_status": verification.get("status"),
        "verification_confidence": verification.get("confidence"),
        "verification_reason": verification.get("reason"),
        "sources": composed.get("sources", []),
        "plan_steps": len(plan),
        "tool_calls": len(trace["tool_results"]),
        "retries": retry_count,
        "learning_example_id": learning_example_id,
        "duration_s": duration_s,
        "trace_path": str(trace_path),
        "trace": trace,
    }


def render_agent_answer(result: dict[str, Any], show_trace: bool = False) -> str:
    lines = []

    lines.append("")
    lines.append("LocalDocLens Agentic RAG Answer")
    lines.append("")
    lines.append(f"Question: {result['question']}")
    lines.append(f"Intent: {result['intent']['intent']}")
    lines.append(f"Answer mode: {result['answer_mode']}")
    lines.append(f"Verification: {result['verification_status']} ({result['verification_confidence']})")
    lines.append(f"Reason: {result['verification_reason']}")
    lines.append(f"Retries: {result['retries']}")
    lines.append(f"Duration: {result['duration_s']}s")
    lines.append("")
    lines.append("Answer:")
    lines.append(result["answer"])
    lines.append("")

    if result.get("sources"):
        lines.append("Sources:")

        for source in result["sources"]:
            lines.append(f"- {source.get('file_name')} page {source.get('page_number')}")

        lines.append("")

    lines.append(f"Trace saved: {result['trace_path']}")

    if show_trace:
        lines.append("")
        lines.append("Agent plan/tool trace:")

        for step in result["trace"].get("tool_results", []):
            lines.append("")
            lines.append(f"- Step: {step.get('step_id')}")
            lines.append(f"  Agent: {step.get('agent')}")
            lines.append(f"  Tool: {step.get('tool')}")
            lines.append(f"  Objective: {step.get('objective')}")

            output = step.get("output")

            if isinstance(output, list):
                lines.append(f"  Output items: {len(output)}")
            else:
                lines.append(f"  Output: {json.dumps(safe_json(output), ensure_ascii=False)[:500]}")

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_agentic_rag(
        question="Should I approve this supplier?",
        file_name=None,
        use_llm=False,
    )
    print(render_agent_answer(result, show_trace=True))
