import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from localdoc.batch_resumable import run_resumable_batch_analysis
from localdoc.stress_eval import evaluate_stress_report


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def safe_read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_warm_api(base_url: str = "http://127.0.0.1:8000") -> dict[str, Any]:
    health_url = base_url.rstrip("/") + "/health"

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(health_url, timeout=3) as response:
            body = response.read().decode("utf-8")
            latency_s = round(time.perf_counter() - start, 4)

        return {
            "available": True,
            "latency_s": latency_s,
            "health_raw": body,
        }

    except Exception as exc:
        return {
            "available": False,
            "latency_s": round(time.perf_counter() - start, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def get_learning_stats() -> dict[str, Any]:
    try:
        from localdoc.learning import LearningExampleStore

        store = LearningExampleStore()
        return store.stats()

    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def get_analysis_db_stats() -> dict[str, Any]:
    try:
        from localdoc.batch_analyze import AnalysisMemoryStore

        store = AnalysisMemoryStore()
        return store.stats()

    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def get_cache_stats() -> dict[str, Any]:
    try:
        from localdoc.cache import AnswerCache
        from localdoc.config import CACHE_DB_PATH

        cache = AnswerCache(CACHE_DB_PATH)
        return cache.stats()

    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_batch_force_and_resume_benchmark(
    file_name: str,
    mode: str,
    force_output_dir: str,
    resume_output_dir: str,
) -> dict[str, Any]:
    force_start = time.perf_counter()

    force_report = run_resumable_batch_analysis(
        file_name=file_name,
        mode=mode,
        use_llm=False,
        output_dir=force_output_dir,
        force=True,
        resume=True,
    )

    force_wall_s = round(time.perf_counter() - force_start, 4)

    resume_start = time.perf_counter()

    resume_report = run_resumable_batch_analysis(
        file_name=file_name,
        mode=mode,
        use_llm=False,
        output_dir=resume_output_dir,
        force=False,
        resume=True,
    )

    resume_wall_s = round(time.perf_counter() - resume_start, 4)

    speedup = None

    if resume_wall_s > 0:
        speedup = round(force_wall_s / resume_wall_s, 2)

    return {
        "file_name": file_name,
        "mode": mode,
        "force_wall_s": force_wall_s,
        "resume_wall_s": resume_wall_s,
        "resume_speedup_x": speedup,
        "force_report": {
            "duration_s": force_report.get("duration_s"),
            "num_files": force_report.get("num_files"),
            "num_pages": force_report.get("num_pages"),
            "num_chunks": force_report.get("num_chunks"),
            "findings": force_report.get("summary", {}).get("num_findings"),
            "high_findings": force_report.get("summary", {}).get("num_high_findings"),
            "job_stats": force_report.get("job_stats", {}),
            "output_files": force_report.get("output_files", {}),
        },
        "resume_report": {
            "duration_s": resume_report.get("duration_s"),
            "num_files": resume_report.get("num_files"),
            "num_pages": resume_report.get("num_pages"),
            "num_chunks": resume_report.get("num_chunks"),
            "findings": resume_report.get("summary", {}).get("num_findings"),
            "high_findings": resume_report.get("summary", {}).get("num_high_findings"),
            "job_stats": resume_report.get("job_stats", {}),
            "output_files": resume_report.get("output_files", {}),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = []

    lines.append("# LocalDocLens Benchmark Suite Report")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append(f"Benchmark file: `{report['file_name']}`")
    lines.append(f"Mode: `{report['mode']}`")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Passed: **{report['passed']}**")
    lines.append(f"- Pages analyzed: **{report['batch_benchmark']['force_report']['num_pages']}**")
    lines.append(f"- Chunks analyzed: **{report['batch_benchmark']['force_report']['num_chunks']}**")
    lines.append(f"- Force rebuild wall time: **{report['batch_benchmark']['force_wall_s']}s**")
    lines.append(f"- Resume wall time: **{report['batch_benchmark']['resume_wall_s']}s**")
    lines.append(f"- Resume speedup: **{report['batch_benchmark']['resume_speedup_x']}x**")
    lines.append("")

    resume_stats = report["batch_benchmark"]["resume_report"]["job_stats"]

    lines.append("## Resume / Reuse")
    lines.append("")
    lines.append(f"- Processed pages on resume: **{resume_stats.get('processed_pages')}**")
    lines.append(f"- Skipped/reused pages on resume: **{resume_stats.get('skipped_pages')}**")
    lines.append(f"- Processed documents on resume: **{resume_stats.get('processed_documents')}**")
    lines.append(f"- Skipped/reused documents on resume: **{resume_stats.get('skipped_documents')}**")
    lines.append(f"- Failed items: **{resume_stats.get('failed_items')}**")
    lines.append("")

    evaluation = report["stress_evaluation"]

    lines.append("## Stress-Test Accuracy")
    lines.append("")
    lines.append(f"- Expected risk: **{evaluation.get('expected_overall_risk')}**")
    lines.append(f"- Actual risk: **{evaluation.get('actual_overall_risk')}**")
    lines.append(f"- Overall risk correct: **{evaluation.get('overall_risk_correct')}**")
    lines.append(f"- Expected findings: **{evaluation.get('expected_findings_count')}**")
    lines.append(f"- Actual findings: **{evaluation.get('actual_findings_count')}**")
    lines.append(f"- Exact type+page matches: **{evaluation.get('exact_type_and_page_matches')}**")
    lines.append(f"- Missing expected findings: **{evaluation.get('missing_expected_findings')}**")
    lines.append(f"- Wrong-page matches: **{evaluation.get('wrong_page_matches')}**")
    lines.append(f"- False positives: **{evaluation.get('false_positives')}**")
    lines.append(f"- Precision exact type+page: **{evaluation.get('precision_exact_type_and_page')}**")
    lines.append(f"- Recall exact type+page: **{evaluation.get('recall_exact_type_and_page')}**")
    lines.append("")

    lines.append("## Learning / Memory / Cache")
    lines.append("")
    lines.append("### Learning examples")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["learning_stats"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Analysis memory")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["analysis_db_stats"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Answer cache")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["cache_stats"], indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Warm API Check")
    lines.append("")
    if report["warm_api"]["available"]:
        lines.append(f"- API available: **True**")
        lines.append(f"- Health latency: **{report['warm_api']['latency_s']}s**")
    else:
        lines.append("- API available: **False**")
        lines.append(f"- Reason: `{report['warm_api'].get('error')}`")
        lines.append("")
        lines.append("This is okay if the server was not running during the benchmark.")
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    for key, value in report["output_files"].items():
        lines.append(f"- {key}: `{value}`")

    return "\n".join(lines)


def run_benchmark_suite(
    file_name: str = "stress_1000_supplier_packet.pdf",
    ground_truth: str = "artifacts/stress_1000_ground_truth.json",
    mode: str = "hybrid",
    output_dir: str = "artifacts/benchmark_suite",
    max_false_positives: int = 1,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    force_output_dir = str(output_path / "force_rebuild")
    resume_output_dir = str(output_path / "resume_reuse")

    batch_benchmark = run_batch_force_and_resume_benchmark(
        file_name=file_name,
        mode=mode,
        force_output_dir=force_output_dir,
        resume_output_dir=resume_output_dir,
    )

    resume_report_path = Path(resume_output_dir) / "batch_analysis_report.json"
    stress_eval_path = output_path / "stress_eval_report.json"

    stress_evaluation = evaluate_stress_report(
        report_path=str(resume_report_path),
        ground_truth_path=ground_truth,
        output_path=str(stress_eval_path),
    )

    warm_api = check_warm_api()
    learning_stats = get_learning_stats()
    analysis_db_stats = get_analysis_db_stats()
    cache_stats = get_cache_stats()

    resume_stats = batch_benchmark["resume_report"]["job_stats"]

    resume_ok = (
        int(resume_stats.get("processed_pages", -1)) == 0
        and int(resume_stats.get("skipped_pages", 0)) >= int(batch_benchmark["resume_report"]["num_pages"])
        and int(resume_stats.get("skipped_documents", 0)) >= 1
    )

    accuracy_ok = (
        stress_evaluation.get("overall_risk_correct") is True
        and int(stress_evaluation.get("missing_expected_findings", 999)) == 0
        and int(stress_evaluation.get("wrong_page_matches", 999)) == 0
        and int(stress_evaluation.get("false_positives", 999)) <= max_false_positives
    )

    report = {
        "generated_at": utc_now(),
        "file_name": file_name,
        "ground_truth": ground_truth,
        "mode": mode,
        "passed": bool(resume_ok and accuracy_ok),
        "resume_ok": bool(resume_ok),
        "accuracy_ok": bool(accuracy_ok),
        "max_false_positives_allowed": max_false_positives,
        "batch_benchmark": batch_benchmark,
        "stress_evaluation": stress_evaluation,
        "warm_api": warm_api,
        "learning_stats": learning_stats,
        "analysis_db_stats": analysis_db_stats,
        "cache_stats": cache_stats,
    }

    json_path = output_path / "benchmark_suite_report.json"
    md_path = output_path / "benchmark_suite_report.md"

    report["output_files"] = {
        "benchmark_json": str(json_path),
        "benchmark_markdown": str(md_path),
        "stress_eval_json": str(stress_eval_path),
        "force_batch_json": str(Path(force_output_dir) / "batch_analysis_report.json"),
        "resume_batch_json": str(resume_report_path),
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    return report


if __name__ == "__main__":
    result = run_benchmark_suite()
    print(json.dumps(result, indent=2, ensure_ascii=False))
