import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


console = Console()


HELP_TEXT = """
[bold]How to attach PDFs or folders[/bold]

Paste a full file path or folder path.

[bold]Single PDF examples:[/bold]
C:\\Users\\amogh\\Downloads\\supplier_packet.pdf
"C:\\Users\\amogh\\My Documents\\supplier packet.pdf"

[bold]Folder examples:[/bold]
C:\\Users\\amogh\\Documents\\supplier_pdfs
"C:\\Users\\amogh\\Downloads\\Supplier Packets"

[bold]PowerShell tip:[/bold]
You can drag a PDF or folder into PowerShell and it will paste the full path.

[bold]Commands:[/bold]
!help   Show this help message
!quit   Exit guided startup
"""


def is_port_in_use(host: str = "127.0.0.1", port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def localdoc_health_ok() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def normalize_path(raw_path: str) -> Path:
    value = raw_path.strip()

    if value.startswith("& "):
        value = value[2:].strip()

    value = value.strip('"').strip("'").strip()

    value = os.path.expanduser(value)
    value = os.path.expandvars(value)

    return Path(value)


def show_help():
    console.print(Panel(HELP_TEXT.strip(), title="LocalDocLens Path Help", border_style="cyan"))


def discover_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise RuntimeError(f"File is not a PDF: {path}")

        return [path]

    if path.is_dir():
        pdfs = sorted(path.rglob("*.pdf"))

        if not pdfs:
            raise RuntimeError(f"No PDF files found in folder: {path}")

        return pdfs

    raise RuntimeError(f"Path does not exist: {path}")


def get_pdf_page_count(path: Path) -> int | None:
    if PdfReader is None:
        return None

    try:
        reader = PdfReader(str(path))

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return None

        return len(reader.pages)
    except Exception:
        return None


def estimate_processing_seconds(num_pages: int | None, num_files: int, total_mb: float) -> int:
    """
    Rough local estimate.

    Native-text PDFs are faster.
    Scanned OCR PDFs are slower.
    Since we do not know scanned/native status until extraction starts,
    this gives a conservative estimate.
    """
    if num_pages and num_pages > 0:
        return max(10, int(num_pages * 2.0))

    return max(10, int(total_mb * 3.0) + (num_files * 3))


def copy_pdfs_to_data_docs(pdfs: list[Path], docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Copying PDFs into data/docs", total=len(pdfs))

        for pdf in pdfs:
            destination = docs_dir / pdf.name

            if destination.exists():
                source_size = pdf.stat().st_size
                dest_size = destination.stat().st_size

                if source_size == dest_size:
                    copied.append(destination)
                    progress.advance(task)
                    continue

                stem = pdf.stem
                suffix = pdf.suffix
                counter = 2

                while destination.exists():
                    destination = docs_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(pdf, destination)
            copied.append(destination)
            progress.advance(task)

    return copied


def localdoc_base_command() -> list[str]:
    executable = shutil.which("localdoc")

    if executable:
        return [executable]

    return [sys.executable, "-m", "localdoc.cli"]


def run_command_with_estimated_progress(command: list[str], description: str, estimated_seconds: int) -> int:
    start_time = time.perf_counter()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    recent_lines = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=estimated_seconds)

        while True:
            line = process.stdout.readline() if process.stdout else ""

            if line:
                stripped = line.rstrip()

                if stripped:
                    recent_lines.append(stripped)
                    recent_lines = recent_lines[-6:]

            return_code = process.poll()
            elapsed = time.perf_counter() - start_time

            progress.update(task, completed=min(elapsed, estimated_seconds))

            if return_code is not None:
                break

            time.sleep(0.15)

        if progress.tasks[0].completed < estimated_seconds:
            progress.update(task, completed=estimated_seconds)

    if recent_lines:
        table = Table(title="Recent processing output")
        table.add_column("Line", overflow="fold")

        for line in recent_lines:
            table.add_row(line)

        console.print(table)

    return process.returncode if process.returncode is not None else 1


def prompt_for_path() -> Path:
    console.print(
        Panel(
            "[bold cyan]Paste a PDF file path or a folder path containing PDFs.[/bold cyan]\n\n"
            "Type [bold]!help[/bold] for examples or [bold]!quit[/bold] to exit.",
            title="LocalDocLens Guided Startup",
            border_style="green",
        )
    )

    while True:
        raw = console.input("[bold]PDF/folder path> [/bold]").strip()

        if not raw:
            continue

        if raw.lower() == "!help":
            show_help()
            continue

        if raw.lower() in {"!quit", "quit", "exit"}:
            raise KeyboardInterrupt

        path = normalize_path(raw)

        if not path.exists():
            console.print(f"[red]Path not found:[/red] {path}")
            console.print("Type [bold]!help[/bold] if you need path examples.")
            continue

        try:
            pdfs = discover_pdfs(path)
        except Exception as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        console.print(f"[green]Found {len(pdfs)} PDF file(s).[/green]")
        return path


def summarize_inputs(pdfs: list[Path]):
    total_bytes = sum(pdf.stat().st_size for pdf in pdfs)
    total_mb = total_bytes / (1024 * 1024)

    page_counts = []
    unknown_pages = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Estimating pages", total=len(pdfs))

        for pdf in pdfs:
            count = get_pdf_page_count(pdf)

            if count is None:
                unknown_pages += 1
            else:
                page_counts.append(count)

            progress.advance(task)

    total_known_pages = sum(page_counts)
    total_pages = total_known_pages if unknown_pages == 0 else None

    table = Table(title="Document Batch")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("PDF files", str(len(pdfs)))
    table.add_row("Total size", f"{total_mb:.2f} MB")

    if total_pages is None:
        table.add_row("Pages", f"{total_known_pages}+ known, {unknown_pages} file(s) unknown")
    else:
        table.add_row("Pages", str(total_pages))

    console.print(table)

    estimated_seconds = estimate_processing_seconds(
        num_pages=total_pages or total_known_pages,
        num_files=len(pdfs),
        total_mb=total_mb,
    )

    return {
        "num_files": len(pdfs),
        "total_mb": total_mb,
        "known_pages": total_known_pages,
        "unknown_page_files": unknown_pages,
        "estimated_seconds": estimated_seconds,
    }


def start_guided(input_path: str = "", serve: bool = True):
    docs_dir = Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            "[bold]LocalDocLens Product Startup[/bold]\n\n"
            "This guided mode will attach PDFs, analyze them, build indexes, "
            "refresh automatic supplier memory, and then start the local API server.",
            title="LocalDocLens",
            border_style="cyan",
        )
    )

    if input_path.strip():
        path = normalize_path(input_path)

        if not path.exists():
            raise RuntimeError(f"Path not found: {path}")
    else:
        path = prompt_for_path()

    pdfs = discover_pdfs(path)
    stats = summarize_inputs(pdfs)

    console.print(
        Panel(
            f"[bold]Estimated processing time:[/bold] about {stats['estimated_seconds']} seconds\n\n"
            "This is a rough estimate. OCR-heavy scanned PDFs can take longer. "
            "Native text PDFs are usually faster.",
            title="Estimate",
            border_style="yellow",
        )
    )

    copied = copy_pdfs_to_data_docs(pdfs, docs_dir)

    console.print(f"[green]Attached {len(copied)} PDF(s) into {docs_dir}.[/green]")

    ingest_command = localdoc_base_command() + ["ingest", str(docs_dir)]

    console.print("")
    console.print("[bold cyan]Starting document analysis pipeline...[/bold cyan]")
    console.print("This runs extraction/OCR, chunking, embedding, and index storage.")

    return_code = run_command_with_estimated_progress(
        command=ingest_command,
        description="Analyzing documents and building indexes",
        estimated_seconds=stats["estimated_seconds"],
    )

    if return_code != 0:
        raise RuntimeError(
            "Document analysis failed. Run localdoc ingest data/docs manually to see full logs."
        )

    console.print("[green]Document analysis completed.[/green]")

    if not serve:
        console.print("[yellow]Skipping server startup because --no-serve was provided.[/yellow]")
        return

    console.print("")
    console.print("[bold cyan]Starting LocalDocLens server...[/bold cyan]")
    console.print("Automatic supplier memory will refresh during server startup.")

    if is_port_in_use("127.0.0.1", 8000):
        if localdoc_health_ok():
            console.print(
                Panel(
                    "[green]LocalDocLens server already appears to be running.[/green]\n\n"
                    "URL: http://127.0.0.1:8000\n"
                    "Health: http://127.0.0.1:8000/health\n\n"
                    "You can open another PowerShell window and ask questions now.",
                    title="Server Already Running",
                    border_style="green",
                )
            )
            return

        console.print(
            Panel(
                "[red]Port 8000 is already in use by another process.[/red]\n\n"
                "Run this in PowerShell to free it:\n\n"
                "$portProcess = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue\n"
                "$processId = $portProcess.OwningProcess\n"
                "Stop-Process -Id $processId -Force",
                title="Port Conflict",
                border_style="red",
            )
        )
        return

    console.print("Press Ctrl+C to stop the server.")

    serve_command = localdoc_base_command() + ["serve"]

    try:
        subprocess.run(serve_command, check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]LocalDocLens server stopped.[/yellow]")


if __name__ == "__main__":
    start_guided()
