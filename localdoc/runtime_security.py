import hmac
import os
from pathlib import Path
from typing import Iterable

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


LOOPBACK_HOSTS = {
    "127.0.0.1",
    "::1",
    "localhost",
}


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_api_token() -> str:
    return os.getenv("LOCALDOCLENS_API_TOKEN", "").strip()


def remote_access_allowed() -> bool:
    return env_truthy("LOCALDOCLENS_ALLOW_REMOTE", default=False)


def is_loopback_client(host: str | None) -> bool:
    if not host:
        return True

    host = host.strip().lower()

    if host in LOOPBACK_HOSTS:
        return True

    if host.startswith("127."):
        return True

    return False


def get_allowed_origins() -> list[str]:
    raw = os.getenv(
        "LOCALDOCLENS_ALLOWED_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:8000,http://localhost:8000",
    )

    origins = [item.strip() for item in raw.split(",") if item.strip()]

    # Never allow wildcard CORS by default.
    if "*" in origins and not env_truthy("LOCALDOCLENS_ALLOW_WILDCARD_CORS", default=False):
        origins = [origin for origin in origins if origin != "*"]

    return origins


def install_cors(app):
    origins = get_allowed_origins()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-LocalDoc-Token"],
        expose_headers=[],
        max_age=600,
    )


def install_runtime_security(app):
    @app.middleware("http")
    async def localdoc_runtime_security(request: Request, call_next):
        client_host = request.client.host if request.client else None

        if not remote_access_allowed() and not is_loopback_client(client_host):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Remote access is disabled. LocalDocLens only accepts localhost requests by default."
                },
            )

        api_token = get_api_token()

        if api_token and request.url.path not in {"/health", "/docs", "/openapi.json"}:
            supplied_token = request.headers.get("X-LocalDoc-Token", "")

            if not hmac.compare_digest(supplied_token, api_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid X-LocalDoc-Token header."},
                )

        response = await call_next(request)

        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

        return response


def validate_safe_file_name(file_name: str) -> str:
    """
    Allow only simple file names, not paths.

    Good:
      supplier.pdf

    Bad:
      ../supplier.pdf
      C:\\Users\\x\\secret.pdf
      folder/supplier.pdf
    """
    if not file_name:
        raise ValueError("file_name cannot be empty")

    candidate = Path(file_name)

    if candidate.name != file_name:
        raise ValueError("file_name must be a simple file name, not a path")

    if ".." in candidate.parts:
        raise ValueError("file_name cannot contain path traversal")

    if candidate.suffix.lower() != ".pdf":
        raise ValueError("file_name must be a PDF file")

    return candidate.name


def safe_path_under(base_dir: str | Path, user_path: str | Path) -> Path:
    base = Path(base_dir).resolve()
    target = (base / user_path).resolve()

    if base != target and base not in target.parents:
        raise ValueError("Unsafe path: path escapes allowed directory")

    return target


def redact_for_log(value: str, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").split())

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text
