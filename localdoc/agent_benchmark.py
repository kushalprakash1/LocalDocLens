import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from localdoc.agentic_rag import run_agentic_rag


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def contains_all_terms(answer: str, expected_terms: list[str]) -> bool:
    answer_norm = normalize(answer)

    for term in expected_terms:
        if normalize(term) not in answer_norm:
            return False

    return True


def source_pages(result: dict[str, Any]) -> set[int]:
    pages = set()

    for source in result.get("sources", []):
        try:
            pages.add(int(source.get("page_number")))
        except Exception:
            pass

    return pages


def run_one_case(case: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    result = run_agentic_rag(
        question=case["question"],
        file_name=case.get("file"),
        top_k=case.get("top_k", 5),
        use_llm=case.get("use_llm", False),
        max_retries=case.get("max_retries", 1),
        output_dir=case.get("output_dir", "artifacts/agent_benchmark/traces"),
    )

    wall_s = round(time.perf_counter() - start, 4)

    expected_terms = case.get("expected_terms", [])
    expected_pages = set(case.get("expected_pages", []))

    actual_pages = source_pages(result)

    terms_hit = contains_all_terms(result.get("answer", ""), expected_terms)
    pages_hit = expected_pages.issubset(actual_pages) if expected_pages else True
    verified = result.get("verification_status") == "auto_verified"

    passed = bool(terms_hit and pages_hit and verified)

    return {
        "id": case["id"],
        "question": case["question"],
        "file": case.get("file"),
        "passed": passed,
        "terms_hit": terms_hit,
        "pages_hit": pages_hit,
        "verified": verified,
        "expected_terms": expected_terms,
        "expected_pages": sorted(expected_pages),
        "actual_pages": sorted(actual_pages),
        "verification_status": result.get("verification_status"),
        "verification_confidence": result.get("verification_confidence"),
        "intent": result.get("intent", {}).get("intent"),
        "answer_mode": result.get("answer_mode"),
        "retries": result.get("retries"),
        "duration_s": result.get("duration_s"),
        "wall_s": wall_s,
        "answer": result.get("answer"),
        "trace_path": result.get("trace_path"),
    }


def default_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "stress_approval_decision",
            "question": "Should I approve this supplier?",
            "file": "stress_1000_supplier_packet.pdf",
            "expected_terms": ["High risk", "Do not approve", "expired insurance", "missing W-9", "missing supplier agreement signature"],
            "expected_pages": [25, 250, 750],
        },
        {
            "id": "stress_insurance_expired",
            "question": "Is the insurance expired?",
            "file": "stress_1000_supplier_packet.pdf",
            "expected_terms": ["expired", "03/12/2025"],
            "expected_pages": [250],
        },
        {
            "id": "stress_bank_name",
            "question": "What is the bank name?",
            "file": "stress_1000_supplier_packet.pdf",
            "expected_terms": ["Central Valley Business Bank"],
            "expected_pages": [500],
        },
        {
            "id": "aurora_bank_name",
            "question": "What is the bank name?",
            "file": "aurora_grain_supplier_packet.pdf",
            "expected_terms": ["Golden State Business Bank"],
            "expected_pages": [3],
        },
        {
            "id": "aurora_payment_terms",
            "question": "What are the payment terms?",
            "file": "aurora_grain_supplier_packet.pdf",
            "expected_terms": ["Net 30"],
            "expected_pages": [3],
        },
    ]


def run_agent_benchmark(
    output_dir: str = "artifacts/agent_benchmark",
    use_llm: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = default_cases()

    for case in cases:
        case["use_llm"] = use_llm
        case["output_dir"] = str(output_path / "traces")

    start = time.perf_counter()
    results = []

    for case in cases:
        results.append(run_one_case(case))

    duration_s = round(time.perf_counter() - start, 4)

    passed_count = len([item for item in results if item["passed"]])
    verified_count = len([item for item in results if item["verified"]])
    pages_hit_count = len([item for item in results if item["pages_hit"]])
    terms_hit_count = len([item for item in results if item["terms_hit"]])

    report = {
        "generated_at": utc_now(),
        "use_llm": use_llm,
        "passed": passed_count == len(results),
        "num_cases": len(results),
        "passed_cases": passed_count,
        "verified_cases": verified_count,
        "terms_hit_cases": terms_hit_count,
        "pages_hit_cases": pages_hit_count,
        "pass_rate": round(passed_count / max(1, len(results)), 4),
        "verification_rate": round(verified_count / max(1, len(results)), 4),
        "source_page_hit_rate": round(pages_hit_count / max(1, len(results)), 4),
        "term_hit_rate": round(terms_hit_count / max(1, len(results)), 4),
        "duration_s": duration_s,
        "results": results,
    }

    json_path = output_path / "agent_benchmark_report.json"
    md_path = output_path / "agent_benchmark_report.md"

    report["output_files"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = []

    lines.append("# LocalDocLens Agentic RAG Benchmark")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append(f"Use LLM: {report['use_llm']}")
    lines.append(f"Passed: **{report['passed']}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases: {report['num_cases']}")
    lines.append(f"- Passed cases: {report['passed_cases']}")
    lines.append(f"- Pass rate: {report['pass_rate']}")
    lines.append(f"- Verification rate: {report['verification_rate']}")
    lines.append(f"- Source page hit rate: {report['source_page_hit_rate']}")
    lines.append(f"- Term hit rate: {report['term_hit_rate']}")
    lines.append(f"- Duration: {report['duration_s']}s")
    lines.append("")
    lines.append("## Cases")
    lines.append("")

    for item in report["results"]:
        lines.append(f"### {item['id']}")
        lines.append("")
        lines.append(f"- Passed: **{item['passed']}**")
        lines.append(f"- Question: {item['question']}")
        lines.append(f"- File: {item['file']}")
        lines.append(f"- Intent: {item['intent']}")
        lines.append(f"- Verification: {item['verification_status']} ({item['verification_confidence']})")
        lines.append(f"- Expected pages: {item['expected_pages']}")
        lines.append(f"- Actual pages: {item['actual_pages']}")
        lines.append(f"- Terms hit: {item['terms_hit']}")
        lines.append(f"- Pages hit: {item['pages_hit']}")
        lines.append(f"- Retries: {item['retries']}")
        lines.append(f"- Duration: {item['duration_s']}s")
        lines.append(f"- Trace: `{item['trace_path']}`")
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append("```text")
        lines.append(str(item["answer"]))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def render_summary(report: dict[str, Any]) -> str:
    lines = []

    lines.append("")
    lines.append("LocalDocLens agentic RAG benchmark completed.")
    lines.append(f"Passed: {report['passed']}")
    lines.append(f"Cases: {report['num_cases']}")
    lines.append(f"Passed cases: {report['passed_cases']}")
    lines.append(f"Pass rate: {report['pass_rate']}")
    lines.append(f"Verification rate: {report['verification_rate']}")
    lines.append(f"Source page hit rate: {report['source_page_hit_rate']}")
    lines.append(f"Term hit rate: {report['term_hit_rate']}")
    lines.append(f"Duration: {report['duration_s']}s")
    lines.append("")
    lines.append("Cases:")

    for item in report["results"]:
        lines.append(f"- {item['id']}: passed={item['passed']}, verified={item['verified']}, pages_hit={item['pages_hit']}")

    lines.append("")
    lines.append("Saved:")
    lines.append(report["output_files"]["json"])
    lines.append(report["output_files"]["markdown"])

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_agent_benchmark()
    print(render_summary(result))
