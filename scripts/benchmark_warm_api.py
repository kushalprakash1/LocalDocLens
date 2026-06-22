import csv
import json
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "eval_questions.json"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "http://127.0.0.1:8000/ask"


def answer_hit(answer: str, expected_answer: str) -> bool:
    return expected_answer.lower() in answer.lower()


def page_hit(response_json: dict, expected_page: int) -> bool:
    pages = [int(item["page_number"]) for item in response_json.get("evidence", [])]
    return int(expected_page) in pages


def main():
    eval_items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    print("\n=== LocalDocLens Warm API Benchmark ===\n")

    results = []

    # Warm-up call
    requests.post(
        API_URL,
        json={
            "question": "warm up retrieval",
            "top_k": 3,
        },
        timeout=60,
    )

    for item in eval_items:
        payload = {
            "question": item["question"],
            "top_k": 6,
        }

        start = time.perf_counter()
        response = requests.post(API_URL, json=payload, timeout=120)
        wall_latency = time.perf_counter() - start

        response.raise_for_status()
        data = response.json()

        answer = data["answer"]

        a_hit = answer_hit(answer, item["expected_answer"])
        p_hit = page_hit(data, item["expected_page"])
        strict = a_hit and p_hit

        row = {
            "id": item["id"],
            "question": item["question"],
            "wall_latency_s": round(wall_latency, 4),
            "server_latency_s": data["latency_s"],
            "retrieval_latency_s": data["retrieval_latency_s"],
            "response_chars": len(answer),
            "expected_answer": item["expected_answer"],
            "expected_page": item["expected_page"],
            "answer_hit": a_hit,
            "page_hit": p_hit,
            "strict_hit": strict,
            "answer": answer,
        }

        results.append(row)

        print(f"- {item['id']}")
        print(f"  wall latency: {row['wall_latency_s']}s")
        print(f"  server latency: {row['server_latency_s']}s")
        print(f"  retrieval latency: {row['retrieval_latency_s']}s")
        print(f"  answer hit: {row['answer_hit']}")
        print(f"  page hit: {row['page_hit']}")
        print(f"  strict hit: {row['strict_hit']}")
        print(f"  answer: {answer}")
        print()

    latencies = [row["wall_latency_s"] for row in results]
    server_latencies = [row["server_latency_s"] for row in results]
    retrieval_latencies = [row["retrieval_latency_s"] for row in results]
    strict_hits = [row["strict_hit"] for row in results]
    answer_hits = [row["answer_hit"] for row in results]
    page_hits = [row["page_hit"] for row in results]

    summary = {
        "num_questions": len(results),
        "avg_wall_latency_s": round(sum(latencies) / len(latencies), 4),
        "min_wall_latency_s": round(min(latencies), 4),
        "max_wall_latency_s": round(max(latencies), 4),
        "avg_server_latency_s": round(sum(server_latencies) / len(server_latencies), 4),
        "avg_retrieval_latency_s": round(sum(retrieval_latencies) / len(retrieval_latencies), 4),
        "answer_hit_accuracy": round(sum(answer_hits) / len(answer_hits), 3),
        "page_hit_accuracy": round(sum(page_hits) / len(page_hits), 3),
        "strict_accuracy_answer_and_page": round(sum(strict_hits) / len(strict_hits), 3),
        "avg_response_chars": round(sum(row["response_chars"] for row in results) / len(results), 2),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    json_path = ARTIFACTS_DIR / "warm_api_benchmark_report.json"
    csv_path = ARTIFACTS_DIR / "warm_api_benchmark_answers.csv"

    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_fields = [
        "id",
        "question",
        "wall_latency_s",
        "server_latency_s",
        "retrieval_latency_s",
        "response_chars",
        "expected_answer",
        "expected_page",
        "answer_hit",
        "page_hit",
        "strict_hit",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for row in results:
            writer.writerow({field: row[field] for field in csv_fields})

    print("=== Warm API Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
