import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PATH = ROOT / "data" / "eval_questions.json"


def bytes_to_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 3)


def path_size(path: Path) -> int:
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                pass
    return total


def localdoc_exe() -> str:
    win_exe = ROOT / ".venv" / "Scripts" / "localdoc.exe"
    if win_exe.exists():
        return str(win_exe)
    return "localdoc"


def run_command(args: list[str]) -> dict:
    start = time.perf_counter()

    try:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ok = proc.returncode == 0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except Exception as e:
        ok = False
        stdout = ""
        stderr = repr(e)
        returncode = -1

    end = time.perf_counter()

    return {
        "ok": ok,
        "returncode": returncode,
        "latency_s": round(end - start, 4),
        "stdout": stdout,
        "stderr": stderr,
        "combined_output": stdout + "\n" + stderr,
    }


def extract_pages(text: str) -> list[int]:
    pages = set()

    patterns = [
        r"\bPage[:\s]+(\d+)\b",
        r"\bpage[:\s]+(\d+)\b",
        r"::p(\d+)::c\d+",
        r"\bp(\d+)::c\d+",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                pages.add(int(match))
            except ValueError:
                pass

    return sorted(pages)


def answer_metrics(output: str, expected_answer: str, expected_page, expected_terms: list[str]) -> dict:
    output_lower = output.lower()

    answer_hit = expected_answer.lower() in output_lower if expected_answer else None

    pages_found = extract_pages(output)
    if expected_page is None:
        page_hit = None
    else:
        page_hit = int(expected_page) in pages_found

    term_hits = []
    for term in expected_terms:
        hit = term.lower() in output_lower
        term_hits.append(
            {
                "term": term,
                "hit": hit,
            }
        )

    if expected_terms:
        term_coverage = sum(1 for item in term_hits if item["hit"]) / len(expected_terms)
    else:
        term_coverage = None

    strict_hit = bool(answer_hit)
    if expected_page is not None:
        strict_hit = strict_hit and bool(page_hit)

    return {
        "answer_hit": answer_hit,
        "page_hit": page_hit,
        "pages_found": pages_found,
        "term_hits": term_hits,
        "term_coverage": round(term_coverage, 3) if term_coverage is not None else None,
        "strict_hit": strict_hit,
    }


def get_ollama_list() -> str:
    result = run_command(["ollama", "list"])
    if result["ok"]:
        return result["stdout"]
    return result["stderr"]


def main() -> None:
    skip_ingest = "--skip-ingest" in sys.argv

    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing eval file: {EVAL_PATH}")

    eval_items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "skip_ingest": skip_ingest,
        "sizes_mb": {},
        "ingest": None,
        "answers": [],
        "summary": {},
        "ollama_list": "",
    }

    print("\n=== LocalDocLens Benchmark ===\n")

    if not skip_ingest:
        print("Running ingestion benchmark...")
        ingest_result = run_command([localdoc_exe(), "ingest", "data/docs"])
        report["ingest"] = {
            "ok": ingest_result["ok"],
            "returncode": ingest_result["returncode"],
            "latency_s": ingest_result["latency_s"],
            "stdout_tail": ingest_result["stdout"][-2000:],
            "stderr_tail": ingest_result["stderr"][-2000:],
        }
        print(f"Ingest latency: {ingest_result['latency_s']}s")
        print(f"Ingest ok: {ingest_result['ok']}")
        print()

    print("Running answer benchmarks...")

    for item in eval_items:
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_page = item.get("expected_page")
        expected_terms = item.get("expected_terms", [])

        result = run_command([localdoc_exe(), "ask", question, "--llm"])
        output = result["combined_output"]

        metrics = answer_metrics(
            output=output,
            expected_answer=expected_answer,
            expected_page=expected_page,
            expected_terms=expected_terms,
        )

        response_chars = len(output)
        response_bytes = len(output.encode("utf-8", errors="replace"))
        estimated_tokens = round(response_chars / 4)

        row = {
            "id": item["id"],
            "question": question,
            "ok": result["ok"],
            "latency_s": result["latency_s"],
            "response_chars": response_chars,
            "response_bytes": response_bytes,
            "estimated_tokens": estimated_tokens,
            "expected_answer": expected_answer,
            "expected_page": expected_page,
            **metrics,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
        }

        report["answers"].append(row)

        print(f"- {item['id']}")
        print(f"  latency: {result['latency_s']}s")
        print(f"  response chars: {response_chars}")
        print(f"  answer hit: {metrics['answer_hit']}")
        print(f"  page hit: {metrics['page_hit']}")
        print(f"  pages found: {metrics['pages_found']}")
        print(f"  strict hit: {metrics['strict_hit']}")
        print()

    answer_rows = report["answers"]

    latencies = [row["latency_s"] for row in answer_rows]
    answer_hits = [row["answer_hit"] for row in answer_rows if row["answer_hit"] is not None]
    page_hits = [row["page_hit"] for row in answer_rows if row["page_hit"] is not None]
    strict_hits = [row["strict_hit"] for row in answer_rows]

    report["sizes_mb"] = {
        "docs_folder_mb": bytes_to_mb(path_size(ROOT / "data" / "docs")),
        "lancedb_folder_mb": bytes_to_mb(path_size(ROOT / "data" / "db")),
        "bm25_indexes_folder_mb": bytes_to_mb(path_size(ROOT / "data" / "indexes")),
        "rendered_pages_folder_mb": bytes_to_mb(path_size(ROOT / "data" / "rendered_pages")),
        "artifacts_folder_mb": bytes_to_mb(path_size(ROOT / "artifacts")),
        "venv_folder_mb": bytes_to_mb(path_size(ROOT / ".venv")),
        "project_without_venv_mb": bytes_to_mb(
            path_size(ROOT / "localdoc")
            + path_size(ROOT / "data")
            + path_size(ROOT / "scripts")
            + path_size(ROOT / "pyproject.toml")
        ),
        "huggingface_cache_mb": bytes_to_mb(path_size(Path.home() / ".cache" / "huggingface")),
        "paddleocr_cache_mb": bytes_to_mb(path_size(Path.home() / ".paddleocr")),
    }

    report["ollama_list"] = get_ollama_list()

    report["summary"] = {
        "num_questions": len(answer_rows),
        "avg_answer_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "min_answer_latency_s": min(latencies) if latencies else None,
        "max_answer_latency_s": max(latencies) if latencies else None,
        "answer_hit_accuracy": round(sum(answer_hits) / len(answer_hits), 3) if answer_hits else None,
        "page_hit_accuracy": round(sum(page_hits) / len(page_hits), 3) if page_hits else None,
        "strict_accuracy_answer_and_page": round(sum(strict_hits) / len(strict_hits), 3) if strict_hits else None,
        "avg_response_chars": round(sum(row["response_chars"] for row in answer_rows) / len(answer_rows), 2) if answer_rows else None,
        "avg_response_bytes": round(sum(row["response_bytes"] for row in answer_rows) / len(answer_rows), 2) if answer_rows else None,
        "avg_estimated_tokens": round(sum(row["estimated_tokens"] for row in answer_rows) / len(answer_rows), 2) if answer_rows else None,
    }

    json_path = ARTIFACTS_DIR / "benchmark_report.json"
    csv_path = ARTIFACTS_DIR / "benchmark_answers.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_fields = [
        "id",
        "question",
        "ok",
        "latency_s",
        "response_chars",
        "response_bytes",
        "estimated_tokens",
        "expected_answer",
        "expected_page",
        "answer_hit",
        "page_hit",
        "pages_found",
        "term_coverage",
        "strict_hit",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for row in answer_rows:
            csv_row = {field: row.get(field) for field in csv_fields}
            csv_row["pages_found"] = json.dumps(row.get("pages_found", []))
            writer.writerow(csv_row)

    print("\n=== Summary ===")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")

    print("\n=== Sizes MB ===")
    for key, value in report["sizes_mb"].items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()