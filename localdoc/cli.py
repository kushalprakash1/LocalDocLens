from pathlib import Path
import typer
from rich.console import Console


from localdoc.answering.qwen_answer import answer_with_qwen
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
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    console.print("[green]LocalDocLens initialized.[/green]")
    console.print(f"Put PDFs inside: {DOCS_DIR}")


@app.command()
def ingest(path: str = typer.Argument(..., help="Folder containing PDFs")):
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
def ask(
    query: str = typer.Argument(..., help="Question to ask your documents"),
    llm: bool = typer.Option(False, "--llm", help="Use local Qwen through Ollama"),
):
    """
    Search documents using BM25 + semantic embeddings.
    Optionally answer using local Qwen.
    """
    console.print(f"[cyan]Searching for:[/cyan] {query}")

    retriever = HybridRetriever()
    results = retriever.search(query)

    if llm:
        console.print("[cyan]Generating answer with local Qwen...[/cyan]")
        answer = answer_with_qwen(query, results)
    else:
        answer = format_evidence_answer(query, results)

    console.print(answer)




@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """
    Start the LocalDocLens warm API server.

    This keeps the embedder, LanceDB table, and BM25 index loaded in memory
    so repeated answers are much faster than cold CLI calls.
    """
    import uvicorn

    uvicorn.run(
        "localdoc.server:app",
        host=host,
        port=port,
        reload=reload,
    )




@app.command()
def report(
    output_dir: str = "artifacts",
    file: str = "",
):
    """
    Generate a structured supplier compliance report from the indexed packet.

    Use --file when multiple PDFs are indexed so evidence from different suppliers
    does not get mixed into one report.
    """
    from localdoc.report import build_supplier_report

    selected_file = file.strip() or None

    try:
        result = build_supplier_report(output_dir=output_dir, file_name=selected_file)
    except Exception as exc:
        print("")
        print("Could not generate report.")
        print(str(exc))
        raise typer.Exit(code=1)

    print("")
    print("LocalDocLens supplier compliance report generated.")
    print(f"Supplier: {result['supplier']}")
    print(f"Source File: {result['source_file']}")
    print(f"Overall Risk: {result['overall_risk']}")
    print(f"Decision: {result['decision']}")
    print("")
    print("Saved:")
    print(result["output_files"]["markdown"])
    print(result["output_files"]["json"])


@app.command()
def inspect(
    output_dir: str = "artifacts",
):
    """
    Inspect indexed PDFs, pages, chunks, OCR confidence, and answer-cache counts.
    """
    from localdoc.inspect import inspect_index

    result = inspect_index(output_dir=output_dir)

    print("")
    print("LocalDocLens index inspection")
    print(f"Total files: {result['total_files']}")
    print(f"Total chunks: {result['total_chunks']}")
    print("")

    for item in result["files"]:
        print(f"- {item['file_name']}")
        print(f"  pages: {item['pages']}")
        print(f"  chunks: {item['chunks']}")
        print(f"  extraction methods: {item['extraction_methods']}")
        print(f"  avg OCR confidence: {item['avg_ocr_confidence']}")
        print("")

    cache = result["cache"]

    print("Cache:")
    print(f"  total cached answers: {cache['total_cached_answers']}")
    print(f"  verified cached answers: {cache['verified_cached_answers']}")
    print(f"  total cache hits: {cache['total_cache_hits']}")
    print("")
    print("Saved:")
    print(result["output_file"])


@app.command()
def extract(
    file: str = typer.Option(..., "--file", "-f", help="Indexed PDF file name to extract text from."),
    page: int = typer.Option(0, "--page", "-p", help="Optional page number to extract. Use 0 for all pages."),
    output_dir: str = typer.Option("artifacts", "--output-dir", help="Directory to save extracted text outputs."),
):
    """
    Show the raw text extracted from an indexed PDF.

    Use this to prove LocalDocLens is reading the uploaded PDF instead of using
    hard-coded supplier data.
    """
    from localdoc.extract import extract_file_text

    try:
        result = extract_file_text(
            file_name=file,
            page=page,
            output_dir=output_dir,
        )
    except Exception as exc:
        print("")
        print("Could not extract indexed text.")
        print(str(exc))
        raise typer.Exit(code=1)

    print("")
    print("LocalDocLens extracted text")
    print(f"File: {result['file_name']}")
    print(f"Pages: {[item['page_number'] for item in result['pages']]}")
    print("")

    for item in result["pages"]:
        print("=" * 80)
        print(f"PAGE {item['page_number']}")
        print(f"Chunks: {item['chunks']}")
        print(f"Extraction methods: {item['extraction_methods']}")
        print(f"Average OCR confidence: {item['avg_ocr_confidence']}")
        print("=" * 80)
        print(item["text"][:2000])

        if len(item["text"]) > 2000:
            print("... text clipped in terminal preview; full text saved to file ...")

        print("")

    print("Saved:")
    print(result["output_files"]["text"])
    print(result["output_files"]["json"])


@app.command()
def facts(
    file: str = typer.Option("", "--file", "-f", help="Optional indexed PDF file name. If omitted, extracts facts for all indexed files."),
    output_dir: str = typer.Option("artifacts", "--output-dir", help="Directory to save extracted fact outputs."),
):
    """
    Extract reusable supplier facts from indexed PDFs.

    This creates a fact layer so simple questions can be answered without
    calling the local LLM every time.
    """
    from localdoc.facts import extract_supplier_facts

    selected_file = file.strip() or None

    try:
        result = extract_supplier_facts(
            file_name=selected_file,
            output_dir=output_dir,
        )
    except Exception as exc:
        print("")
        print("Could not extract supplier facts.")
        print(str(exc))
        raise typer.Exit(code=1)

    print("")
    print("LocalDocLens supplier facts extracted.")
    print(f"Files processed: {result['num_files']}")
    print("")

    for facts in result["facts"]:
        supplier = facts["supplier_name"]["value"]
        file_name = facts["file_name"]
        risk = facts["risk"]["overall_risk"]
        decision = facts["risk"]["decision"]

        print(f"- {supplier or file_name}")
        print(f"  file: {file_name}")
        print(f"  risk: {risk}")
        print(f"  decision: {decision}")
        print(f"  email: {facts['email']['value']}")
        print(f"  bank: {facts['bank']['bank_name']['value']}")
        print(f"  payment terms: {facts['bank']['payment_terms']['value']}")
        print(f"  policy number: {facts['insurance']['policy_number']['value']}")
        print(f"  agreement status: {facts['agreement']['status']['value']}")
        print("")

    print("Saved:")
    print(result["output_files"]["combined_json"])
    print(result["output_files"]["combined_markdown"])

    for path in result["output_files"]["per_file_json"]:
        print(path)


@app.command()
def start(
    path: str = typer.Option("", "--path", "-p", help="Optional PDF file path or folder path. If omitted, LocalDocLens asks interactively."),
    no_serve: bool = typer.Option(False, "--no-serve", help="Analyze documents but do not start the server."),
):
    """
    Guided product startup.

    Prompts the user for a PDF/folder path, attaches PDFs into data/docs,
    runs ingestion, refreshes indexes/memory, and starts the local API server.
    """
    from localdoc.start import start_guided

    try:
        start_guided(
            input_path=path,
            serve=not no_serve,
        )
    except KeyboardInterrupt:
        print("")
        print("LocalDocLens startup cancelled.")
        raise typer.Exit(code=0)
    except Exception as exc:
        print("")
        print("LocalDocLens startup failed.")
        print(str(exc))
        raise typer.Exit(code=1)


@app.command("export-training-data")
def export_training_data(
    output: str = typer.Option("artifacts/training_examples.jsonl", "--output", "-o", help="Path to save exported training/evaluation examples."),
    include_needs_review: bool = typer.Option(False, "--include-needs-review", help="Include examples marked needs_review."),
    include_unverified: bool = typer.Option(False, "--include-unverified", help="Include unverified examples."),
):
    """
    Export locally collected learning examples as JSONL.

    By default, only auto_verified examples are exported.
    """
    from localdoc.learning import LearningExampleStore

    statuses = ["auto_verified"]

    if include_needs_review:
        statuses.append("needs_review")

    if include_unverified:
        statuses.append("unverified")

    store = LearningExampleStore()
    result = store.export_jsonl(
        output_path=output,
        include_statuses=statuses,
    )

    print("")
    print("LocalDocLens training/evaluation examples exported.")
    print(f"Output: {result['output_path']}")
    print(f"Examples exported: {result['num_examples_exported']}")
    print(f"Included statuses: {', '.join(result['included_statuses'])}")


@app.command("batch-analyze")
def batch_analyze(
    file: str = typer.Option("", "--file", "-f", help="Optional indexed PDF file name. If omitted, analyzes all indexed PDFs."),
    mode: str = typer.Option("hybrid", "--mode", help="Analysis mode: fast, hybrid, or llm."),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use local Qwen for document summaries. Slower but more expressive."),
    output_dir: str = typer.Option("artifacts", "--output-dir", help="Directory to save batch analysis outputs."),
    force: bool = typer.Option(False, "--force", help="Force rebuild page/document memory instead of reusing existing records."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Disable resumable reuse and process everything again."),
):
    """
    Build resumable page memory, document memory, findings, and batch risk summary.

    This is the scalable offline analysis layer for large PDF batches.
    """
    from localdoc.batch_resumable import run_resumable_batch_analysis

    selected_file = file.strip() or None

    try:
        report = run_resumable_batch_analysis(
            file_name=selected_file,
            mode=mode,
            use_llm=use_llm,
            output_dir=output_dir,
            force=force,
            resume=not no_resume,
        )
    except Exception as exc:
        print("")
        print("LocalDocLens batch analysis failed.")
        print(str(exc))
        raise typer.Exit(code=1)

    stats = report["job_stats"]

    print("")
    print("LocalDocLens resumable batch analysis completed.")
    print(f"Job ID: {report['job_id']}")
    print(f"Mode: {report['mode']}")
    print(f"Use LLM: {report['use_llm']}")
    print(f"Resume enabled: {report['resume']}")
    print(f"Force rebuild: {report['force']}")
    print(f"Files analyzed: {report['num_files']}")
    print(f"Pages analyzed: {report['num_pages']}")
    print(f"Chunks analyzed: {report['num_chunks']}")
    print(f"Findings: {report['summary']['num_findings']}")
    print(f"High findings: {report['summary']['num_high_findings']}")
    print(f"Duration: {report['duration_s']}s")
    print("")
    print("Resume stats:")
    print(f"- processed pages: {stats.get('processed_pages')}")
    print(f"- skipped/reused pages: {stats.get('skipped_pages')}")
    print(f"- processed documents: {stats.get('processed_documents')}")
    print(f"- skipped/reused documents: {stats.get('skipped_documents')}")
    print(f"- failed items: {stats.get('failed_items')}")
    print("")
    print("Risk distribution:")

    for risk, count in sorted(report["summary"]["risk_distribution"].items()):
        print(f"- {risk}: {count}")

    print("")
    print("Saved:")
    print(report["output_files"]["json"])
    print(report["output_files"]["markdown"])
    print(report["analysis_db_path"])


@app.command("stress-test")
def stress_test(
    pages: int = typer.Option(1000, "--pages", help="Number of pages to generate."),
    output: str = typer.Option("data/docs/stress_1000_supplier_packet.pdf", "--output", "-o", help="PDF output path."),
    risk_profile: str = typer.Option("high", "--risk-profile", help="Risk profile: high or clean."),
    supplier_name: str = typer.Option("MegaScale Ingredients LLC", "--supplier-name", help="Synthetic supplier legal name."),
    ground_truth_output: str = typer.Option("artifacts/stress_1000_ground_truth.json", "--ground-truth-output", help="Ground truth JSON output path."),
):
    """
    Generate a synthetic large supplier PDF for stress testing.

    This creates a controlled native-text PDF and a ground-truth JSON file.
    """
    from localdoc.stress_test import generate_stress_pdf

    try:
        result = generate_stress_pdf(
            pages=pages,
            output=output,
            risk_profile=risk_profile,
            supplier_name=supplier_name,
            ground_truth_output=ground_truth_output,
        )
    except Exception as exc:
        print("")
        print("LocalDocLens stress-test generation failed.")
        print(str(exc))
        raise typer.Exit(code=1)

    print("")
    print("LocalDocLens synthetic stress-test PDF generated.")
    print(f"PDF: {result['pdf_path']}")
    print(f"Ground truth: {result['ground_truth_path']}")
    print(f"Pages: {result['pages']}")
    print(f"Supplier: {result['supplier_name']}")
    print(f"Risk profile: {result['risk_profile']}")
    print(f"Expected risk: {result['expected_overall_risk']}")
    print(f"Expected findings: {result['expected_findings']}")


@app.command("stress-eval")
def stress_eval(
    report: str = typer.Option("artifacts/stress_1000_clean_v2/batch_analysis_report.json", "--report", help="Batch analysis report JSON path."),
    ground_truth: str = typer.Option("artifacts/stress_1000_ground_truth.json", "--ground-truth", help="Ground truth JSON path."),
    output: str = typer.Option("artifacts/stress_eval_report.json", "--output", "-o", help="Evaluation output JSON path."),
):
    """
    Evaluate a synthetic stress-test report against its ground truth.
    """
    from localdoc.stress_eval import evaluate_stress_report, render_summary

    try:
        result = evaluate_stress_report(
            report_path=report,
            ground_truth_path=ground_truth,
            output_path=output,
        )
    except Exception as exc:
        print("")
        print("LocalDocLens stress evaluation failed.")
        print(str(exc))
        raise typer.Exit(code=1)

    print(render_summary(result))
    print("")
    print("Saved:")
    print(output)


@app.command("benchmark-suite")
def benchmark_suite(
    file: str = typer.Option("stress_1000_supplier_packet.pdf", "--file", "-f", help="Indexed stress-test PDF file name."),
    ground_truth: str = typer.Option("artifacts/stress_1000_ground_truth.json", "--ground-truth", help="Stress-test ground truth JSON path."),
    mode: str = typer.Option("hybrid", "--mode", help="Benchmark mode: fast, hybrid, or llm."),
    output_dir: str = typer.Option("artifacts/benchmark_suite", "--output-dir", help="Directory to save benchmark suite outputs."),
    max_false_positives: int = typer.Option(1, "--max-false-positives", help="Maximum allowed false positives for pass/fail."),
):
    """
    Run the LocalDocLens benchmark suite.

    This measures batch analysis, resume/reuse, stress-test accuracy,
    learning database stats, cache stats, and optional warm API health.
    """
    from localdoc.benchmark_suite import run_benchmark_suite

    try:
        report = run_benchmark_suite(
            file_name=file,
            ground_truth=ground_truth,
            mode=mode,
            output_dir=output_dir,
            max_false_positives=max_false_positives,
        )
    except Exception as exc:
        print("")
        print("LocalDocLens benchmark suite failed.")
        print(str(exc))
        print("")
        print("Most common fix:")
        print("1. Make sure the stress PDF exists in data/docs.")
        print("2. Run: localdoc ingest data/docs")
        print("3. Run benchmark-suite again.")
        raise typer.Exit(code=1)

    batch = report["batch_benchmark"]
    eval_result = report["stress_evaluation"]
    resume_stats = batch["resume_report"]["job_stats"]

    print("")
    print("LocalDocLens benchmark suite completed.")
    print(f"Passed: {report['passed']}")
    print("")
    print("Batch/resume:")
    print(f"- force rebuild wall time: {batch['force_wall_s']}s")
    print(f"- resume wall time: {batch['resume_wall_s']}s")
    print(f"- resume speedup: {batch['resume_speedup_x']}x")
    print(f"- processed pages on resume: {resume_stats.get('processed_pages')}")
    print(f"- skipped/reused pages on resume: {resume_stats.get('skipped_pages')}")
    print(f"- processed documents on resume: {resume_stats.get('processed_documents')}")
    print(f"- skipped/reused documents on resume: {resume_stats.get('skipped_documents')}")
    print("")
    print("Stress accuracy:")
    print(f"- expected risk: {eval_result.get('expected_overall_risk')}")
    print(f"- actual risk: {eval_result.get('actual_overall_risk')}")
    print(f"- risk correct: {eval_result.get('overall_risk_correct')}")
    print(f"- expected findings: {eval_result.get('expected_findings_count')}")
    print(f"- actual findings: {eval_result.get('actual_findings_count')}")
    print(f"- exact type+page matches: {eval_result.get('exact_type_and_page_matches')}")
    print(f"- false positives: {eval_result.get('false_positives')}")
    print(f"- precision: {eval_result.get('precision_exact_type_and_page')}")
    print(f"- recall: {eval_result.get('recall_exact_type_and_page')}")
    print("")
    print("Saved:")
    for key, value in report["output_files"].items():
        print(f"- {key}: {value}")


@app.command("security-check")
def security_check(
    root: str = typer.Option(".", "--root", help="Project root to scan."),
):
    """
    Scan the project for common open-source release security/privacy risks.
    """
    from localdoc.security_check import run_security_check, render_security_summary

    result = run_security_check(root=root)
    print(render_security_summary(result))

    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command("security-endpoint-test")
def security_endpoint_test(
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url", help="LocalDocLens API base URL."),
    token: str = typer.Option("", "--token", help="Optional X-LocalDoc-Token value."),
):
    """
    Test API endpoint hardening against unsafe file path inputs.
    """
    from localdoc.security_endpoint_test import run_security_endpoint_test, render_summary

    result = run_security_endpoint_test(
        base_url=base_url,
        token=token,
    )

    print(render_summary(result))

    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command("agent-ask")
def agent_ask(
    question: str = typer.Argument(..., help="Question to ask the agentic RAG system."),
    file: str = typer.Option("", "--file", "-f", help="Optional indexed PDF file name."),
    top_k: int = typer.Option(5, "--top-k", help="Evidence chunks per retrieval step."),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use local Qwen to compose the final answer."),
    max_retries: int = typer.Option(1, "--max-retries", help="Verifier retry attempts if evidence support is weak."),
    output_dir: str = typer.Option("artifacts/agent_runs", "--output-dir", help="Directory to save agent traces."),
    show_trace: bool = typer.Option(False, "--show-trace", help="Print the agent plan/tool trace."),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON output."),
):
    """
    Ask a question using the full agentic RAG workflow.

    Planner -> Memory -> Retriever -> Reasoner -> Verifier -> Retry -> Trace
    """
    from localdoc.agentic_rag import run_agentic_rag, render_agent_answer
    from localdoc.runtime_security import validate_safe_file_name

    selected_file = file.strip() or None

    if selected_file:
        selected_file = validate_safe_file_name(selected_file)

    try:
        result = run_agentic_rag(
            question=question,
            file_name=selected_file,
            top_k=top_k,
            use_llm=use_llm,
            max_retries=max_retries,
            output_dir=output_dir,
        )
    except Exception as exc:
        print("")
        print("LocalDocLens agentic RAG failed.")
        print(str(exc))
        raise typer.Exit(code=1)

    if json_output:
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_agent_answer(result, show_trace=show_trace))


@app.command("agent-benchmark")
def agent_benchmark(
    output_dir: str = typer.Option("artifacts/agent_benchmark", "--output-dir", help="Directory to save agent benchmark outputs."),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use local Qwen for final answer composition during benchmark."),
):
    """
    Benchmark the full agentic RAG workflow.

    Tests planning, memory, retrieval, reasoning, verification, source-page correctness, and trace saving.
    """
    from localdoc.agent_benchmark import run_agent_benchmark, render_summary

    try:
        report = run_agent_benchmark(
            output_dir=output_dir,
            use_llm=use_llm,
        )
    except Exception as exc:
        print("")
        print("LocalDocLens agent benchmark failed.")
        print(str(exc))
        print("")
        print("Most common fix:")
        print("1. Make sure stress_1000_supplier_packet.pdf exists.")
        print("2. Run: localdoc ingest data/docs")
        print("3. Run: localdoc agent-benchmark")
        raise typer.Exit(code=1)

    print(render_summary(report))

    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("release-check")
def release_check(
    skip_pip_audit: bool = typer.Option(False, "--skip-pip-audit", help="Skip dependency vulnerability audit."),
):
    """
    Run final pre-GitHub release checks.

    Checks compile status, security scan, .gitignore, tracked sensitive files,
    required docs, and dependency vulnerabilities.
    """
    from localdoc.release_check import run_release_check, render_summary

    report = run_release_check(skip_pip_audit=skip_pip_audit)
    print(render_summary(report))

    if not report["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
