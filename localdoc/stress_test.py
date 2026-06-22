import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def wrap_text(text: str, width: int = 92) -> list[str]:
    words = str(text).split()
    lines = []
    current = []

    for word in words:
        test = " ".join(current + [word])

        if len(test) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def page_template(
    supplier_name: str,
    page_number: int,
    total_pages: int,
    risk_profile: str,
) -> tuple[str, list[str], dict[str, Any]]:
    """
    Returns title, lines, and ground-truth page metadata.
    """

    metadata: dict[str, Any] = {
        "page_number": page_number,
        "document_type": "supporting_document",
        "contains_key_fact": False,
        "expected_findings": [],
        "expected_fields": {},
    }

    title = f"{supplier_name} - Large Supplier Packet - Page {page_number}"

    # Core supplier profile.
    if page_number == 1:
        metadata["document_type"] = "supplier_profile"
        metadata["contains_key_fact"] = True
        metadata["expected_fields"] = {
            "supplier_name": supplier_name,
            "primary_contact": "Anika Rao",
            "email": "anika.rao@example-megascale.test",
            "phone": "209-555-0148",
            "supplier_type": "Food ingredient supplier",
        }

        lines = [
            "Supplier Profile and Onboarding Summary",
            f"Supplier legal name: {supplier_name}",
            "DBA name: MegaScale Ingredients",
            "Primary contact: Anika Rao",
            "Email: anika.rao@example-megascale.test",
            "Phone: 209-555-0148",
            "Supplier type: Food ingredient supplier",
            "Onboarding checklist owner: Procurement Operations",
            "This synthetic page is generated for LocalDocLens scale testing.",
        ]

        return title, lines, metadata

    # W-9 page.
    if page_number == 25:
        metadata["document_type"] = "w9_tax_form"
        metadata["contains_key_fact"] = True

        if risk_profile == "high":
            metadata["expected_findings"].append("missing_w9")
            metadata["expected_fields"]["w9_status"] = "failed"
            lines = [
                "W-9 Tax Form Review",
                f"Supplier legal name: {supplier_name}",
                "W-9 tax form missing.",
                "Reviewer note: Supplier must provide signed W-9 before approval.",
                "Taxpayer identification section: not received.",
            ]
        else:
            metadata["expected_fields"]["w9_status"] = "passed"
            lines = [
                "W-9 Tax Form Review",
                f"Supplier legal name: {supplier_name}",
                "Signed W-9 received.",
                "W-9 tax form received and reviewed by procurement operations.",
                "Taxpayer identification section: complete.",
            ]

        return title, lines, metadata

    # Insurance page.
    if page_number == 250:
        metadata["document_type"] = "certificate_of_insurance"
        metadata["contains_key_fact"] = True

        if risk_profile == "high":
            metadata["expected_findings"].append("expired_insurance")
            metadata["expected_fields"] = {
                "insurance_status": "expired",
                "policy_carrier": "Pacific Shield Insurance",
                "policy_number": "GL-2024-MSI-1000",
                "expiration_date": "03/12/2025",
            }
            lines = [
                "Certificate of Insurance",
                f"Insured supplier: {supplier_name}",
                "Policy carrier: Pacific Shield Insurance",
                "Policy number: GL-2024-MSI-1000",
                "Coverage type: General liability",
                "Policy expiration date: 03/12/2025",
                "Reviewer note: Certificate of insurance appears expired because the expiration date is before the current onboarding review date.",
            ]
        else:
            metadata["expected_fields"] = {
                "insurance_status": "active",
                "policy_carrier": "Pacific Shield Insurance",
                "policy_number": "GL-2028-MSI-1000",
                "expiration_date": "12/31/2028",
            }
            lines = [
                "Certificate of Insurance",
                f"Insured supplier: {supplier_name}",
                "Policy carrier: Pacific Shield Insurance",
                "Policy number: GL-2028-MSI-1000",
                "Coverage type: General liability",
                "Policy expiration date: 12/31/2028",
                "Reviewer note: Certificate of insurance is active and not expired.",
            ]

        return title, lines, metadata

    # Bank verification.
    if page_number == 500:
        metadata["document_type"] = "bank_verification"
        metadata["contains_key_fact"] = True
        metadata["expected_fields"] = {
            "bank_name": "Central Valley Business Bank",
            "routing_number": "000777888",
            "account_ending": "4422",
            "payment_terms": "Net 30",
            "remittance_email": "ap@example-megascale.test",
        }

        lines = [
            "Bank Verification and ACH Details",
            f"Supplier legal name: {supplier_name}",
            "Bank name: Central Valley Business Bank",
            "Routing number: 000777888",
            "Account number: ending in 4422",
            "ACH payment method: enabled after verification.",
            "Bank verification status: verified by bank letter dated 05/14/2027.",
            "Payment terms: Net 30",
            "Remittance email: ap@example-megascale.test",
            "Invoice due date example: 09/01/2027",
        ]

        return title, lines, metadata

    # Supplier agreement.
    if page_number == 750:
        metadata["document_type"] = "supplier_agreement"
        metadata["contains_key_fact"] = True

        if risk_profile == "high":
            metadata["expected_findings"].append("missing_signature")
            metadata["expected_fields"] = {
                "agreement_status": "failed",
                "buyer_signer": "Jordan Lee",
                "supplier_signer": None,
            }

            lines = [
                "Supplier Agreement Signature Page",
                f"Supplier legal name: {supplier_name}",
                "Agreement effective date: 07/01/2027",
                "Buyer representative signature: Signed",
                "Signed by Jordan Lee on 07/01/2027",
                "Supplier representative signature missing.",
                "Reviewer note: Agreement is pending signature from supplier representative.",
            ]
        else:
            metadata["expected_fields"] = {
                "agreement_status": "passed",
                "buyer_signer": "Jordan Lee",
                "supplier_signer": "Anika Rao",
            }

            lines = [
                "Supplier Agreement Signature Page",
                f"Supplier legal name: {supplier_name}",
                "Agreement effective date: 07/01/2027",
                "Agreement status: signed by both parties.",
                "Buyer representative signature: Signed",
                "Signed by Jordan Lee on 07/01/2027",
                "Supplier representative signature: Signed",
                "Signed by Anika Rao on 07/01/2027",
                "Reviewer note: Agreement is complete and includes the required supplier signature.",
            ]

        return title, lines, metadata

    # Supplemental compliance.
    if page_number == 900:
        metadata["document_type"] = "supplemental_compliance"
        metadata["contains_key_fact"] = True
        metadata["expected_fields"] = {
            "food_safety_certificate": "passed",
            "sanctions_screening": "clear",
            "conflict_of_interest": "none disclosed",
        }

        lines = [
            "Supplemental Compliance Review",
            f"Supplier legal name: {supplier_name}",
            "Food safety certificate: received.",
            "Certificate review status: passed.",
            "Sanctions screening: clear.",
            "Conflict of interest disclosure: none disclosed.",
            "Overall supplemental compliance review: passed.",
        ]

        return title, lines, metadata

    # Repeated section marker pages.
    if page_number % 100 == 0:
        metadata["document_type"] = "section_index"
        lines = [
            "Section Index",
            f"Supplier legal name: {supplier_name}",
            f"This page marks a synthetic section boundary for stress testing page {page_number}.",
            "No compliance exception is stated on this page.",
            "This page should be indexed, summarized, embedded, and reused by resumable batch analysis.",
        ]

        return title, lines, metadata

    # Normal filler/supporting pages.
    document_kinds = [
        "Purchase order support",
        "Warehouse receiving note",
        "Quality assurance note",
        "Ingredient specification appendix",
        "Packaging declaration",
        "Logistics support record",
        "Supplier communication log",
        "Procurement review note",
    ]

    kind = document_kinds[page_number % len(document_kinds)]
    metadata["document_type"] = "supporting_document"

    filler_id = f"SUPPORT-{page_number:04d}-{random.randint(1000, 9999)}"

    lines = [
        kind,
        f"Supplier legal name: {supplier_name}",
        f"Supporting document reference: {filler_id}",
        f"This is synthetic supporting content for page {page_number} of {total_pages}.",
        "No W-9 exception, insurance exception, bank exception, sanctions exception, or supplier agreement exception is stated here.",
        "This page exists to stress test extraction, chunking, embedding assignment, page memory, document memory, and resumable batch processing.",
        "The expected system behavior is to store page memory and avoid treating ordinary support pages as high-risk findings.",
    ]

    return title, lines, metadata


def draw_page(pdf: canvas.Canvas, title: str, lines: list[str], page_number: int, total_pages: int):
    width, height = letter

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, height - 55, title)

    pdf.setFont("Helvetica", 10)
    y = height - 85

    for raw_line in lines:
        for line in wrap_text(raw_line, width=95):
            pdf.drawString(50, y, line)
            y -= 15

            if y < 70:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 55

        y -= 5

    pdf.setFont("Helvetica", 8)
    pdf.drawString(50, 35, f"LocalDocLens synthetic stress test page {page_number} of {total_pages}")
    pdf.showPage()


def generate_stress_pdf(
    pages: int = 1000,
    output: str = "data/docs/stress_1000_supplier_packet.pdf",
    risk_profile: str = "high",
    supplier_name: str = "MegaScale Ingredients LLC",
    ground_truth_output: str = "artifacts/stress_1000_ground_truth.json",
) -> dict[str, Any]:
    if pages < 10:
        raise RuntimeError("pages must be at least 10")

    risk_profile = risk_profile.lower().strip()

    if risk_profile not in {"clean", "high"}:
        raise RuntimeError("risk_profile must be one of: clean, high")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    truth_path = Path(ground_truth_output)
    truth_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=letter)

    page_truth = []
    expected_fields: dict[str, Any] = {}
    expected_findings = []

    for page_number in range(1, pages + 1):
        title, lines, metadata = page_template(
            supplier_name=supplier_name,
            page_number=page_number,
            total_pages=pages,
            risk_profile=risk_profile,
        )

        draw_page(pdf, title, lines, page_number, pages)
        page_truth.append(metadata)

        if metadata.get("expected_fields"):
            expected_fields.update(metadata["expected_fields"])

        for finding in metadata.get("expected_findings", []):
            expected_findings.append(
                {
                    "finding": finding,
                    "page_number": page_number,
                }
            )

    pdf.save()

    ground_truth = {
        "generated_at": utc_now(),
        "pdf_path": str(output_path),
        "pages": pages,
        "supplier_name": supplier_name,
        "risk_profile": risk_profile,
        "expected_overall_risk": "High" if risk_profile == "high" else "Low",
        "expected_fields": expected_fields,
        "expected_findings": expected_findings,
        "key_pages": {
            "supplier_profile": 1,
            "w9": 25,
            "insurance": 250,
            "bank": 500,
            "agreement": 750,
            "supplemental_compliance": 900,
        },
        "page_truth": page_truth,
    }

    truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    return {
        "pdf_path": str(output_path),
        "ground_truth_path": str(truth_path),
        "pages": pages,
        "supplier_name": supplier_name,
        "risk_profile": risk_profile,
        "expected_overall_risk": ground_truth["expected_overall_risk"],
        "expected_findings": expected_findings,
    }


if __name__ == "__main__":
    print(json.dumps(generate_stress_pdf(), indent=2))
