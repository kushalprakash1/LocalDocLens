import json
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "http://127.0.0.1:8000/ask"

QUESTION = "Which suppliers have expired insurance?"


def main():
    results = []

    print("\n=== Verified Cache Latency Benchmark ===\n")

    for i in range(20):
        body = {
            "question": QUESTION,
            "require_verified": True,
            "use_cache": True,
        }

        start = time.perf_counter()
        response = requests.post(API_URL, json=body, timeout=30)
        wall_latency = time.perf_counter() - start
        response.raise_for_status()

        data = response.json()

        row = {
            "run": i + 1,
            "wall_latency_s": round(wall_latency, 6),
            "server_latency_s": data["latency_s"],
            "retrieval_latency_s": data["retrieval_latency_s"],
            "cache_status": data["cache_status"],
            "cache_verified": data["cache_verified"],
            "answer": data["answer"],
        }

        results.append(row)

        print(
            f"run {row['run']:02d} | "
            f"wall={row['wall_latency_s']}s | "
            f"server={row['server_latency_s']}s | "
            f"retrieval={row['retrieval_latency_s']}s | "
            f"{row['cache_status']}"
        )

    latencies = [row["wall_latency_s"] for row in results]
    server_latencies = [row["server_latency_s"] for row in results]

    summary = {
        "question": QUESTION,
        "runs": len(results),
        "avg_wall_latency_s": round(sum(latencies) / len(latencies), 6),
        "min_wall_latency_s": round(min(latencies), 6),
        "max_wall_latency_s": round(max(latencies), 6),
        "avg_server_latency_s": round(sum(server_latencies) / len(server_latencies), 6),
        "all_cache_verified": all(row["cache_verified"] for row in results),
        "all_cache_hits": all(row["cache_status"] == "hit_verified" for row in results),
        "all_retrieval_skipped": all(row["retrieval_latency_s"] == 0 for row in results),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out_path = ARTIFACTS_DIR / "verified_cache_latency_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(out_path)


if __name__ == "__main__":
    main()
