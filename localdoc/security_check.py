import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


RISKY_PATH_PARTS = [
    "data/docs",
    "data\\docs",
    "data/db",
    "data\\db",
    "artifacts",
    ".env",
    ".venv",
]

SECRET_PATTERNS = {
    "openai_key": r"sk-[A-Za-z0-9_\-]{20,}",
    "github_token": r"gh[pousr]_[A-Za-z0-9_]{20,}",
    "huggingface_token": r"hf_[A-Za-z0-9]{20,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "generic_api_key_assignment": r"(?i)(api_key|apikey|secret|token|password)\s*=\s*['\"][^'\"]{12,}['\"]",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".env",
    ".example",
    ".ini",
    ".cfg",
}


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def should_skip_path(path: Path) -> bool:
    parts = set(path.parts)

    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    return bool(parts.intersection(skip_dirs))


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    if path.name in {".gitignore", ".env", ".env.example"}:
        return True

    return False


def run_git_ls_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return []

        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    except Exception:
        return []


def scan_tracked_sensitive_files() -> list[dict[str, Any]]:
    tracked = run_git_ls_files()
    findings = []

    for file_path in tracked:
        normalized = file_path.replace("\\", "/").lower()

        risky = False

        if normalized.startswith("data/docs/"):
            risky = True

        if normalized.startswith("data/db/"):
            risky = True

        if normalized.startswith("artifacts/") and not normalized.endswith(".gitkeep"):
            risky = True

        if normalized.endswith((".sqlite", ".sqlite3", ".db")):
            risky = True

        if normalized in {".env"} or normalized.startswith(".env."):
            risky = True

        if risky:
            findings.append(
                {
                    "severity": "high",
                    "type": "tracked_sensitive_file",
                    "path": file_path,
                    "message": "Sensitive/generated local file appears to be tracked by Git.",
                }
            )

    return findings


def scan_secret_patterns(root: Path) -> list[dict[str, Any]]:
    findings = []

    for path in root.rglob("*"):
        if should_skip_path(path):
            continue

        if not path.is_file():
            continue

        if not is_text_file(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for name, pattern in SECRET_PATTERNS.items():
            for match in re.finditer(pattern, text):
                line_number = text[: match.start()].count("\n") + 1

                # Ignore obvious placeholders in example docs.
                line = text.splitlines()[line_number - 1] if line_number <= len(text.splitlines()) else ""

                if "your_" in line.lower() or "example" in line.lower() or "placeholder" in line.lower():
                    continue

                findings.append(
                    {
                        "severity": "high",
                        "type": "possible_secret",
                        "secret_type": name,
                        "path": str(path),
                        "line": line_number,
                        "message": "Possible secret found. Review before pushing to GitHub.",
                    }
                )

    return findings


def scan_server_exposure(root: Path) -> list[dict[str, Any]]:
    findings = []

    for path in root.rglob("*.py"):
        if should_skip_path(path):
            continue

        # Do not flag the scanner itself for example strings inside detection rules.
        if path.name == "security_check.py":
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if "0.0.0.0" in text:
            findings.append(
                {
                    "severity": "medium",
                    "type": "network_exposure",
                    "path": str(path),
                    "message": "Found 0.0.0.0. Make sure default server binding is localhost unless auth is added.",
                }
            )

        if 'allow_origins=["*"]' in text or "allow_origins=['*']" in text:
            findings.append(
                {
                    "severity": "medium",
                    "type": "wildcard_cors",
                    "path": str(path),
                    "message": "Found wildcard CORS. Restrict origins for any non-local deployment.",
                }
            )

    return findings


def scan_local_data_presence(root: Path) -> list[dict[str, Any]]:
    warnings = []

    local_dirs = [
        root / "data" / "docs",
        root / "data" / "db",
        root / "artifacts",
    ]

    for directory in local_dirs:
        if not directory.exists():
            continue

        files = [
            path for path in directory.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]

        if files:
            warnings.append(
                {
                    "severity": "info",
                    "type": "local_generated_data_present",
                    "path": str(directory),
                    "count": len(files),
                    "message": "Local/generated files exist. This is okay locally, but they should not be committed.",
                }
            )

    return warnings


def run_security_check(root: str = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()

    findings = []
    findings.extend(scan_tracked_sensitive_files())
    findings.extend(scan_secret_patterns(root_path))
    findings.extend(scan_server_exposure(root_path))
    findings.extend(scan_local_data_presence(root_path))

    high = [item for item in findings if item["severity"] == "high"]
    medium = [item for item in findings if item["severity"] == "medium"]

    result = {
        "generated_at": utc_now(),
        "root": str(root_path),
        "passed": len(high) == 0,
        "high_count": len(high),
        "medium_count": len(medium),
        "total_findings": len(findings),
        "findings": findings,
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "security_check_report.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def render_security_summary(result: dict[str, Any]) -> str:
    lines = []

    lines.append("")
    lines.append("LocalDocLens security check completed.")
    lines.append(f"Passed: {result['passed']}")
    lines.append(f"High findings: {result['high_count']}")
    lines.append(f"Medium findings: {result['medium_count']}")
    lines.append(f"Total findings: {result['total_findings']}")
    lines.append("")

    if result["findings"]:
        lines.append("Findings:")

        for item in result["findings"]:
            lines.append("")
            lines.append(f"- [{item['severity'].upper()}] {item['type']}")
            lines.append(f"  Path: {item.get('path')}")
            lines.append(f"  Message: {item.get('message')}")

            if item.get("line"):
                lines.append(f"  Line: {item.get('line')}")
    else:
        lines.append("No findings.")

    lines.append("")
    lines.append("Saved:")
    lines.append("artifacts/security_check_report.json")

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_security_check()
    print(render_security_summary(report))
