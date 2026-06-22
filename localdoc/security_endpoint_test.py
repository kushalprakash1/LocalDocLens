import json
import urllib.error
import urllib.request
from datetime import datetime


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def post_json(url: str, payload: dict, token: str = "") -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }

    if token:
        headers["X-LocalDoc-Token"] = token

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def run_security_endpoint_test(
    base_url: str = "http://127.0.0.1:8000",
    token: str = "",
) -> dict:
    ask_url = base_url.rstrip("/") + "/ask"

    tests = [
        {
            "name": "path_traversal_parent_dirs",
            "payload": {
                "question": "What is the bank name?",
                "file": "../../secret.pdf",
                "mode": "auto",
            },
            "expected_reject": True,
        },
        {
            "name": "absolute_windows_path",
            "payload": {
                "question": "What is the bank name?",
                "file": "C:\\\\Users\\\\amogh\\\\Desktop\\\\secret.pdf",
                "mode": "auto",
            },
            "expected_reject": True,
        },
        {
            "name": "non_pdf_file",
            "payload": {
                "question": "What is the bank name?",
                "file": "analysis_memory.sqlite",
                "mode": "auto",
            },
            "expected_reject": True,
        },
        {
            "name": "safe_pdf_file_name",
            "payload": {
                "question": "What is the bank name?",
                "file": "aurora_grain_supplier_packet.pdf",
                "mode": "auto",
                "use_cache": False,
                "use_llm": False,
            },
            "expected_reject": False,
        },
    ]

    results = []

    for test in tests:
        status, body = post_json(ask_url, test["payload"], token=token)

        rejected = status >= 400

        passed = rejected == test["expected_reject"]

        results.append(
            {
                "name": test["name"],
                "status_code": status,
                "expected_reject": test["expected_reject"],
                "actual_rejected": rejected,
                "passed": passed,
                "body_preview": body[:500],
            }
        )

    passed = all(item["passed"] for item in results)

    return {
        "generated_at": utc_now(),
        "base_url": base_url,
        "passed": passed,
        "results": results,
    }


def render_summary(result: dict) -> str:
    lines = []

    lines.append("")
    lines.append("LocalDocLens endpoint security test completed.")
    lines.append(f"Passed: {result['passed']}")
    lines.append("")

    for item in result["results"]:
        lines.append(f"- {item['name']}")
        lines.append(f"  passed: {item['passed']}")
        lines.append(f"  status: {item['status_code']}")
        lines.append(f"  expected reject: {item['expected_reject']}")
        lines.append(f"  actual rejected: {item['actual_rejected']}")

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_security_endpoint_test()
    print(render_summary(result))
