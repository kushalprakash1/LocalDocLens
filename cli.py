from pathlib import Path
import typer
from rich.console import Console

from localdoc.ingestion.pdf_loader import load_pdf_pages
from localdoc.indexing.chunker import make_page_chunks
from localdoc.indexing.embedder import Embedder
from localdoc.indexing.vector_store import VectorStore
from localdoc.indexing.bm25_store import BM25Store
from localdoc.retrieval.hybrid_retriever import HybridRetriever
from localdoc.answering.evidence_answer import format_evidence_answer
from localdoc.config import DOCS_DIR

app = typer.Typer()
console = Console()


@app.command()
def init():
    """
    Initialize local project folders.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    console.print("[green]LocalDocLens initialized.[/green]")
    console.print(f"Put PDFs inside: {DOCS_DIR}")


@app.command()
def ingest(path: str = typer.Argument(..., help="Folder containing PDFs")):
    """
    Ingest PDFs and build vector + BM25 indexes.
    """
    folder = Path(path)

    if not folder.exists():
        raise typer.BadParameter(f"Folder does not exist: {folder}")

    pdfs = list(folder.glob("*.pdf"))

    if not pdfs:
        console.print("[red]No PDFs found.[/red]")
        return

    console.print(f"[cyan]Found {len(pdfs)} PDFs.[/cyan]")

    all_chunks = []

    for pdf in pdfs:
        console.print(f"[yellow]Reading {pdf.name}...[/yellow]")
        pages = load_pdf_pages(pdf)

        for page in pages:
            chunks = make_page_chunks(page)
            all_chunks.extend(chunks)

    console.print(f"[cyan]Created {len(all_chunks)} chunks.[/cyan]")

    if not all_chunks:
        console.print("[red]No text chunks created. These PDFs may be scanned images. OCR comes next phase.[/red]")
        return

    console.print("[cyan]Embedding chunks with E5...[/cyan]")
    embedder = Embedder()
    texts = [chunk["text"] for chunk in all_chunks]
    vectors = embedder.embed_passages(texts)

    for chunk, vector in zip(all_chunks, vectors):
        chunk["vector"] = vector
        chunk["embedding_model"] = embedder.model_name

    console.print("[cyan]Saving vectors to LanceDB...[/cyan]")
    vector_store = VectorStore()
    vector_store.reset_table()
    vector_store.add_chunks(all_chunks)

    console.print("[cyan]Building BM25 index...[/cyan]")
    bm25_store = BM25Store()
    bm25_store.build(all_chunks)

    console.print("[green]Ingestion complete.[/green]")
    console.print(f"Chunks indexed: {len(all_chunks)}")


@app.command()
def ask(query: str = typer.Argument(..., help="Question to ask your documents")):
    """
    Search documents using BM25 + semantic embeddings.
    """
    console.print(f"[cyan]Searching for:[/cyan] {query}")

    retriever = HybridRetriever()
    results = retriever.search(query)

    answer = format_evidence_answer(query, results)
    console.print(answer)


if __name__ == "__main__":
    app()