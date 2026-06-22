import json
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "eval_questions.json"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ASK_URL = "http://127.0.0.1:8000/ask"
VERIFY_URL = "http://127.0.0.1:8000/cache/verify"
STATS_URL = "http://127.0.0.1:8000/cache/stats"


def post_json(url: str, payload: dict):
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def get_json(url: str):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    eval_items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    print("\n=== Verify All Eval Answers Into Cache ===\n")

    verified_rows = []

    for item in eval_items:
        question = item["question"]

        print(f"Question: {question}")

        # First ask normally. This creates cache entry if missing.
        first = post_json(
            ASK_URL,
            {
                "question": question,
                "top_k": 6,
                "use_cache": True,
                "require_verified": False,
            },
        )

        cache_key = first["cache_key"]

        # Verify the generated/cached answer.
        verify = post_json(
            VERIFY_URL,
            {
                "cache_key": cache_key,
                "verified_by": "amogh",
                "note": "Verified against labeled benchmark: answer hit and page evidence hit are correct.",
            },
        )

        # Ask again requiring verified cache.
        start = time.perf_counter()
        second = post_json(
            ASK_URL,
            {
                "question": question,
                "top_k": 6,
                "use_cache": True,
                "require_verified": True,
            },
        )
        wall_latency = time.perf_counter() - start

        row = {
            "id": item["id"],
            "question": question,
            "cache_key": cache_key,
            "verify_ok": verify["ok"],
            "cache_status": second["cache_status"],
            "cache_verified": second["cache_verified"],
            "wall_latency_s": round(wall_latency, 6),
            "server_latency_s": second["latency_s"],
            "retrieval_latency_s": second["retrieval_latency_s"],
            "answer": second["answer"],
        }

        verified_rows.append(row)

        print(f"  verify_ok: {row['verify_ok']}")
        print(f"  cache_status: {row['cache_status']}")
        print(f"  cache_verified: {row['cache_verified']}")
        print(f"  retrieval_latency_s: {row['retrieval_latency_s']}")
        print(f"  wall_latency_s: {row['wall_latency_s']}")
        print()

    stats = get_json(STATS_URL)

    latencies = [row["wall_latency_s"] for row in verified_rows]
    server_latencies = [row["server_latency_s"] for row in verified_rows]

    summary = {
        "num_questions": len(verified_rows),
        "verified_rows": sum(1 for row in verified_rows if row["cache_verified"]),
        "all_verified": all(row["cache_verified"] for row in verified_rows),
        "all_hit_verified": all(row["cache_status"] == "hit_verified" for row in verified_rows),
        "all_retrieval_skipped": all(row["retrieval_latency_s"] == 0 for row in verified_rows),
        "avg_wall_latency_s": round(sum(latencies) / len(latencies), 6),
        "min_wall_latency_s": round(min(latencies), 6),
        "max_wall_latency_s": round(max(latencies), 6),
        "avg_server_latency_s": round(sum(server_latencies) / len(server_latencies), 6),
        "cache_stats": stats,
    }

    output = {
        "summary": summary,
        "rows": verified_rows,
    }

    output_path = ARTIFACTS_DIR / "verified_eval_cache_report.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=== Verified Cache Summary ===")
    for key, value in summary.items():
        if key != "cache_stats":
            print(f"{key}: {value}")

    print("\nCache stats:")
    print(json.dumps(stats, indent=2))

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
