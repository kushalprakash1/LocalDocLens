import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DANGEROUS_TRACKED_PREFIXES = [
    "data/docs/",
    "data/db/",
    "artifacts/",
]

DANGEROUS_TRACKED_SUFFIXES = [
    ".sqlite",
    ".sqlite3",
    ".db",
    ".env",
    ".pdf",
]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def run_command(command: list[str], timeout: int = 300) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "ok": result.returncode == 0,
        }

    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }


def compile_python_files() -> dict[str, Any]:
    files = [
        str(path)
        for path in Path("localdoc").rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    failures = []

    for file_path in files:
        result = run_command([sys.executable, "-m", "py_compile", file_path], timeout=60)

        if not result["ok"]:
            failures.append(
                {
                    "file": file_path,
                    "stderr": result["stderr"],
                    "stdout": result["stdout"],
                }
            )

    return {
        "ok": len(failures) == 0,
        "files_checked": len(files),
        "failures": failures,
    }


def git_ls_files() -> list[str]:
    result = run_command(["git", "ls-files"], timeout=30)

    if not result["ok"]:
        return []

    return [line.strip().replace("\\", "/") for line in result["stdout"].splitlines() if line.strip()]


def check_tracked_sensitive_files() -> dict[str, Any]:
    tracked = git_ls_files()
    dangerous = []

    for path in tracked:
        lower = path.lower()

        if lower.endswith(".gitkeep"):
            continue

        if any(lower.startswith(prefix) for prefix in DANGEROUS_TRACKED_PREFIXES):
            dangerous.append(path)
            continue

        if any(lower.endswith(suffix) for suffix in DANGEROUS_TRACKED_SUFFIXES):
            dangerous.append(path)
            continue

        if lower.startswith(".env"):
            dangerous.append(path)

    return {
        "ok": len(dangerous) == 0,
        "dangerous_tracked_files": dangerous,
        "tracked_file_count": len(tracked),
    }


def check_gitignore_exists() -> dict[str, Any]:
    path = Path(".gitignore")

    required_terms = [
        "data/docs/*",
        "data/db/*",
        "artifacts/*",
        ".env",
        ".venv/",
        "*.sqlite",
        "*.db",
    ]

    if not path.exists():
        return {
            "ok": False,
            "missing": required_terms,
            "message": ".gitignore missing.",
        }

    text = path.read_text(encoding="utf-8", errors="ignore")
    missing = [term for term in required_terms if term not in text]

    return {
        "ok": len(missing) == 0,
        "missing": missing,
    }


def run_localdoc_security_check() -> dict[str, Any]:
    result = run_command(["localdoc", "security-check"], timeout=120)

    parsed = None
    report_path = Path("artifacts/security_check_report.json")

    if report_path.exists():
        try:
            parsed = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = None

    return {
        "ok": result["ok"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "parsed_report": parsed,
    }


def run_pip_audit(skip: bool = False) -> dict[str, Any]:
    if skip:
        return {
            "ok": True,
            "skipped": True,
            "message": "pip-audit skipped by user.",
        }

    result = run_command(["pip-audit", "-r", "requirements.txt"], timeout=300)

    return {
        "ok": result["ok"],
        "skipped": False,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def check_required_docs() -> dict[str, Any]:
    required = [
        "README.md",
        "SECURITY.md",
        "PRIVACY.md",
        ".env.example",
        ".gitignore",
    ]

    missing = [path for path in required if not Path(path).exists()]

    return {
        "ok": len(missing) == 0,
        "missing": missing,
    }


def run_release_check(skip_pip_audit: bool = False) -> dict[str, Any]:
    results = {
        "generated_at": utc_now(),
        "checks": {
            "python_compile": compile_python_files(),
            "gitignore": check_gitignore_exists(),
            "tracked_sensitive_files": check_tracked_sensitive_files(),
            "required_docs": check_required_docs(),
            "localdoc_security_check": run_localdoc_security_check(),
            "pip_audit": run_pip_audit(skip=skip_pip_audit),
        },
    }

    results["passed"] = all(check.get("ok") for check in results["checks"].values())

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "release_check_report.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results["output_path"] = str(output_path)

    return results


def render_summary(report: dict[str, Any]) -> str:
    lines = []

    lines.append("")
    lines.append("LocalDocLens release check completed.")
    lines.append(f"Passed: {report['passed']}")
    lines.append("")

    for name, result in report["checks"].items():
        lines.append(f"- {name}: {'PASS' if result.get('ok') else 'FAIL'}")

        if name == "python_compile":
            lines.append(f"  files checked: {result.get('files_checked')}")
            if result.get("failures"):
                for failure in result["failures"][:5]:
                    lines.append(f"  compile failure: {failure['file']}")
                    lines.append(f"  stderr: {failure['stderr'][:300]}")

        if name == "tracked_sensitive_files" and result.get("dangerous_tracked_files"):
            lines.append("  dangerous tracked files:")
            for item in result["dangerous_tracked_files"]:
                lines.append(f"  - {item}")

        if name == "gitignore" and result.get("missing"):
            lines.append("  missing .gitignore entries:")
            for item in result["missing"]:
                lines.append(f"  - {item}")

        if name == "required_docs" and result.get("missing"):
            lines.append("  missing docs:")
            for item in result["missing"]:
                lines.append(f"  - {item}")

        if name == "pip_audit" and not result.get("ok"):
            lines.append("  pip-audit found dependency issues or failed.")
            lines.append("  Review artifacts/release_check_report.json for details.")

    lines.append("")
    lines.append("Saved:")
    lines.append(report["output_path"])

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_release_check()
    print(render_summary(report))
    raise SystemExit(0 if report["passed"] else 1)
