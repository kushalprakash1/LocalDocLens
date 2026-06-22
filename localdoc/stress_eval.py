import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def normalize_finding_type(value: str) -> str:
    value = str(value or "").lower().strip()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")

    aliases = {
        "missing_w_9": "missing_w9",
        "missing_w9": "missing_w9",
        "missing_w_9_tax_form": "missing_w9",
        "w9_missing": "missing_w9",
        "expired_insurance": "expired_insurance",
        "insurance_expired": "expired_insurance",
        "missing_signature": "missing_signature",
        "missing_supplier_signature": "missing_signature",
        "missing_supplier_agreement_signature": "missing_signature",
        "supplier_agreement_signature_missing": "missing_signature",
        "bank_not_verified": "bank_not_verified",
        "sanctions_review": "sanctions_review",
    }

    return aliases.get(value, value)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_report_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []

    for document in report.get("documents", []):
        file_name = document.get("file_name")

        for finding in document.get("findings", []):
            normalized = dict(finding)
            normalized["file_name"] = normalized.get("file_name") or file_name
            normalized["normalized_finding_type"] = normalize_finding_type(
                normalized.get("finding_type", "")
            )

            try:
                normalized["page_number"] = int(normalized.get("page_number"))
            except Exception:
                normalized["page_number"] = None

            findings.append(normalized)

    return findings


def extract_expected_findings(ground_truth: dict[str, Any]) -> list[dict[str, Any]]:
    expected = []

    for item in ground_truth.get("expected_findings", []):
        expected.append(
            {
                "finding": item.get("finding"),
                "normalized_finding_type": normalize_finding_type(item.get("finding")),
                "page_number": int(item.get("page_number")),
            }
        )

    return expected


def match_expected_to_actual(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []

    used_actual_indexes = set()

    for expected_item in expected:
        exact_match = None
        type_only_match = None

        for index, actual_item in enumerate(actual):
            if index in used_actual_indexes:
                continue

            same_type = (
                actual_item.get("normalized_finding_type")
                == expected_item.get("normalized_finding_type")
            )

            same_page = actual_item.get("page_number") == expected_item.get("page_number")

            if same_type and same_page:
                exact_match = (index, actual_item)
                break

            if same_type and type_only_match is None:
                type_only_match = (index, actual_item)

        if exact_match:
            used_actual_indexes.add(exact_match[0])
            results.append(
                {
                    "expected": expected_item,
                    "matched": True,
                    "page_correct": True,
                    "match_type": "exact_type_and_page",
                    "actual": exact_match[1],
                }
            )
        elif type_only_match:
            used_actual_indexes.add(type_only_match[0])
            results.append(
                {
                    "expected": expected_item,
                    "matched": True,
                    "page_correct": False,
                    "match_type": "type_only_wrong_page",
                    "actual": type_only_match[1],
                }
            )
        else:
            results.append(
                {
                    "expected": expected_item,
                    "matched": False,
                    "page_correct": False,
                    "match_type": "missing",
                    "actual": None,
                }
            )

    false_positives = []

    for index, actual_item in enumerate(actual):
        if index not in used_actual_indexes:
            false_positives.append(actual_item)

    return results, false_positives


def evaluate_stress_report(
    report_path: str = "artifacts/stress_1000_clean_v2/batch_analysis_report.json",
    ground_truth_path: str = "artifacts/stress_1000_ground_truth.json",
    output_path: str = "artifacts/stress_eval_report.json",
) -> dict[str, Any]:
    report = load_json(report_path)
    ground_truth = load_json(ground_truth_path)

    actual_findings = extract_report_findings(report)
    expected_findings = extract_expected_findings(ground_truth)

    match_results, false_positives = match_expected_to_actual(
        expected=expected_findings,
        actual=actual_findings,
    )

    expected_risk = ground_truth.get("expected_overall_risk")
    actual_risks = []

    for document in report.get("documents", []):
        actual_risks.append(document.get("overall_risk"))

    actual_risk = actual_risks[0] if len(actual_risks) == 1 else actual_risks

    exact_matches = [item for item in match_results if item["matched"] and item["page_correct"]]
    type_matches = [item for item in match_results if item["matched"]]
    missing = [item for item in match_results if not item["matched"]]
    wrong_page = [item for item in match_results if item["matched"] and not item["page_correct"]]

    precision = 0.0
    recall = 0.0
    page_recall = 0.0

    if actual_findings:
        precision = len(exact_matches) / len(actual_findings)

    if expected_findings:
        recall = len(type_matches) / len(expected_findings)
        page_recall = len(exact_matches) / len(expected_findings)

    evaluation = {
        "generated_at": utc_now(),
        "report_path": str(report_path),
        "ground_truth_path": str(ground_truth_path),
        "expected_overall_risk": expected_risk,
        "actual_overall_risk": actual_risk,
        "overall_risk_correct": actual_risk == expected_risk,
        "expected_findings_count": len(expected_findings),
        "actual_findings_count": len(actual_findings),
        "exact_type_and_page_matches": len(exact_matches),
        "type_matches": len(type_matches),
        "missing_expected_findings": len(missing),
        "wrong_page_matches": len(wrong_page),
        "false_positives": len(false_positives),
        "precision_exact_type_and_page": round(precision, 4),
        "recall_type_only": round(recall, 4),
        "recall_exact_type_and_page": round(page_recall, 4),
        "passed": (
            actual_risk == expected_risk
            and len(missing) == 0
            and len(wrong_page) == 0
            and len(false_positives) <= 1
        ),
        "matches": match_results,
        "false_positive_findings": false_positives,
        "actual_findings": actual_findings,
        "expected_findings": expected_findings,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    return evaluation


def render_summary(evaluation: dict[str, Any]) -> str:
    lines = []

    lines.append("")
    lines.append("LocalDocLens stress-test evaluation completed.")
    lines.append(f"Passed: {evaluation['passed']}")
    lines.append("")
    lines.append("Risk:")
    lines.append(f"- expected: {evaluation['expected_overall_risk']}")
    lines.append(f"- actual: {evaluation['actual_overall_risk']}")
    lines.append(f"- correct: {evaluation['overall_risk_correct']}")
    lines.append("")
    lines.append("Findings:")
    lines.append(f"- expected findings: {evaluation['expected_findings_count']}")
    lines.append(f"- actual findings: {evaluation['actual_findings_count']}")
    lines.append(f"- exact type+page matches: {evaluation['exact_type_and_page_matches']}")
    lines.append(f"- type matches: {evaluation['type_matches']}")
    lines.append(f"- missing expected findings: {evaluation['missing_expected_findings']}")
    lines.append(f"- wrong-page matches: {evaluation['wrong_page_matches']}")
    lines.append(f"- false positives: {evaluation['false_positives']}")
    lines.append("")
    lines.append("Scores:")
    lines.append(f"- precision exact type+page: {evaluation['precision_exact_type_and_page']}")
    lines.append(f"- recall type only: {evaluation['recall_type_only']}")
    lines.append(f"- recall exact type+page: {evaluation['recall_exact_type_and_page']}")

    if evaluation["false_positive_findings"]:
        lines.append("")
        lines.append("False positives:")

        for finding in evaluation["false_positive_findings"][:10]:
            lines.append(
                f"- {finding.get('normalized_finding_type')} page {finding.get('page_number')}: {finding.get('message')}"
            )

    if evaluation["matches"]:
        lines.append("")
        lines.append("Expected finding matches:")

        for item in evaluation["matches"]:
            expected = item["expected"]
            actual = item.get("actual") or {}
            lines.append(
                f"- expected {expected['normalized_finding_type']} page {expected['page_number']} -> "
                f"{item['match_type']} actual page {actual.get('page_number')}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    result = evaluate_stress_report()
    print(render_summary(result))
