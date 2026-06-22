# LocalDocLens

LocalDocLens is a **local-first supplier document intelligence system** for procurement onboarding and supplier compliance review.

It reads supplier PDFs, extracts page-level evidence, builds searchable document memory, and answers supplier compliance questions using local retrieval, structured facts, verification, and optional local LLM reasoning.

The goal is simple:

> Give procurement teams a local, auditable assistant that can inspect supplier onboarding packets, find missing documents, identify expired compliance items, cite source pages, and explain whether a supplier should be approved or flagged for review.

---

## What LocalDocLens Can Do

LocalDocLens can process supplier onboarding documents and answer questions such as:

```powershell
localdoc ask "Which suppliers have expired insurance?" --llm
localdoc ask "Which suppliers are missing a W-9?" --llm
localdoc ask "Which documents are missing signatures?" --llm
localdoc ask "What is the bank verification status for ABC Foods LLC?" --llm
localdoc ask "Should I approve this supplier?" --llm
```

It can also run a full agentic supplier review:

```powershell
localdoc agent-ask "Should I approve this supplier?" --file "aurora_grain_supplier_packet.pdf" --show-trace
```

The system returns answers with source-page evidence instead of unsupported guesses.

---

## Why This Project Exists

Supplier onboarding often requires reviewing many documents manually, including:

* W-9 forms
* insurance certificates
* bank verification forms
* supplier agreements
* food safety documents
* compliance forms
* signatures
* expiration dates
* payment terms

A normal chatbot is not enough for this workflow because supplier approval decisions need evidence, traceability, and privacy.

LocalDocLens is designed around those requirements:

* local-first processing
* source-page citations
* structured supplier facts
* hybrid retrieval
* verification before caching
* agent traces for auditability
* no automatic document upload

---

## Core Features

LocalDocLens currently includes:

* PDF ingestion
* native PDF text extraction
* optional OCR for scanned/image-only PDFs
* page-aware chunking
* page-level document memory
* document-level supplier memory
* structured supplier fact extraction
* E5 embeddings using `intfloat/e5-small-v2`
* LanceDB vector index
* BM25 keyword index
* hybrid retrieval
* local Qwen answer generation through Ollama
* source-page citations
* answer verification
* verified answer cache
* local learning examples
* batch supplier analysis
* resumable batch processing
* stress testing on large PDFs
* benchmark suite
* agentic RAG
* API server
* runtime security checks
* release checks for GitHub safety

---

## Agentic RAG System

LocalDocLens includes a full local agentic RAG workflow.

Instead of only retrieving chunks and generating an answer, the agent performs multiple steps:

1. **Planner Agent**
   Classifies the user question and decides what type of task is being asked.

2. **Memory Agent**
   Loads structured supplier facts from the document memory.

3. **Retriever Agent**
   Retrieves targeted source pages for the relevant compliance items.

4. **Reasoner Agent**
   Builds a supplier answer or approval decision from the facts and evidence.

5. **Verifier Agent**
   Checks whether the answer is supported by the retrieved evidence.

6. **Retry Loop**
   If evidence support is weak, the system can retry with expanded retrieval.

7. **Trace Logger**
   Saves the full agent workflow trace for review.

Example:

```powershell
localdoc agent-ask "Should I approve this supplier?" --file "stress_1000_supplier_packet.pdf" --show-trace
```

Example output summary:

```text
Decision: Do not approve until failed compliance items are resolved.
Failed items: expired insurance, missing W-9, missing supplier agreement signature.
Sources:
- stress_1000_supplier_packet.pdf page 25
- stress_1000_supplier_packet.pdf page 250
- stress_1000_supplier_packet.pdf page 500
- stress_1000_supplier_packet.pdf page 750
```

This makes LocalDocLens more than a basic RAG chatbot. It acts like a supplier-document review agent with evidence and verification.

---

## Local-First Design

LocalDocLens is designed to run on your own machine.

By default:

* PDFs stay in `data/docs`
* databases stay in `data/db`
* generated reports stay in `artifacts`
* embeddings are stored locally
* cached answers are stored locally
* agent traces are stored locally
* no supplier document is automatically uploaded to the project maintainer

If you use Ollama, the local LLM endpoint usually runs at:

```text
http://127.0.0.1:11434
```

You should verify that your model server is local before processing sensitive supplier documents.

---

## Project Structure

Typical project layout:

```text
LocalDocLens/
├── localdoc/
│   ├── cli.py
│   ├── server.py
│   ├── facts.py
│   ├── memory.py
│   ├── cache.py
│   ├── verifier.py
│   ├── learning.py
│   ├── agentic_rag.py
│   ├── agent_benchmark.py
│   ├── batch_analyze.py
│   ├── batch_resumable.py
│   ├── stress_test.py
│   ├── stress_eval.py
│   ├── benchmark_suite.py
│   ├── security_check.py
│   ├── runtime_security.py
│   └── release_check.py
│
├── data/
│   ├── docs/
│   └── db/
│
├── artifacts/
│
├── docs/
│   └── OCR_SETUP.md
│
├── requirements.txt
├── requirements-ocr.txt
├── SECURITY.md
├── PRIVACY.md
├── .gitignore
└── README.md
```

Important local folders:

```text
data/docs      supplier PDFs
data/db        local databases and indexes
artifacts      reports, traces, benchmark outputs
```

These folders are intentionally ignored by Git except for `.gitkeep` placeholder files.

---

## Requirements

Recommended environment:

* Windows 10 or 11
* Python 3.12
* PowerShell
* Git
* Ollama for local LLM answering
* Optional OCR dependencies for scanned PDFs

The default install is the safer core install.

OCR dependencies are separated because OCR packages are heavier and may have separate compatibility or vulnerability advisories.

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/kushalprakash1/LocalDocLens.git
cd LocalDocLens
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install core dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional OCR install:

```powershell
pip install -r requirements-ocr.txt
```

OCR is only needed if you want scanned/image-only PDF support.

---

## Optional OCR Setup

The stable OCR setup used during development was:

```text
Python 3.12
paddlepaddle==2.6.2
paddleocr==2.7.3
numpy==1.26.4
opencv-python==4.8.1.78
```

OCR packages are installed separately through:

```powershell
pip install -r requirements-ocr.txt
```

Do not upgrade PaddleOCR or PaddlePaddle without testing in a separate branch. Newer OCR dependency combinations may cause Windows CPU runtime issues.

More OCR notes are in:

```text
docs/OCR_SETUP.md
```

---

## Ollama Setup for Local LLM Answers

Install Ollama from:

```text
https://ollama.com
```

Pull a local model:

```powershell
ollama pull qwen3:4b
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

Then use LocalDocLens with `--llm`:

```powershell
localdoc ask "Which suppliers have expired insurance?" --llm
```

LocalDocLens can still perform deterministic extraction, retrieval, facts, reports, and agent workflows without relying on external cloud APIs.

---

## Basic Usage

Place supplier PDFs in:

```text
data/docs
```

Ingest documents:

```powershell
localdoc ingest data/docs
```

Ask a question:

```powershell
localdoc ask "Which suppliers have expired insurance?" --llm
```

Ask without local LLM generation:

```powershell
localdoc ask "Which suppliers have expired insurance?"
```

Inspect stored memory:

```powershell
localdoc inspect
```

View extracted facts:

```powershell
localdoc facts
```

Generate a supplier compliance report:

```powershell
localdoc report
```

---

## Agentic Supplier Review

Run an agentic supplier review:

```powershell
localdoc agent-ask "Should I approve this supplier?" --file "aurora_grain_supplier_packet.pdf" --show-trace
```

Use local LLM composition:

```powershell
localdoc agent-ask "Should I approve this supplier?" --file "aurora_grain_supplier_packet.pdf" --use-llm --show-trace
```

Save JSON output:

```powershell
localdoc agent-ask "Should I approve this supplier?" --file "aurora_grain_supplier_packet.pdf" --json
```

Run the agent benchmark:

```powershell
localdoc agent-benchmark
```

Agent traces are saved under:

```text
artifacts/agent_runs
```

Do not commit agent traces if they were generated from real supplier documents.

---

## Warm API Server

LocalDocLens includes a FastAPI server for warm, low-latency operation.

Start the server:

```powershell
localdoc serve
```

Default local address:

```text
http://127.0.0.1:8000
```

The warm server loads the embedder, vector index, and BM25 index once. This avoids repeatedly starting a new Python process for each question.

Typical local flow:

```powershell
localdoc ingest data/docs
localdoc serve
```

Then use the API from another terminal or application.

---

## API Security

The server is designed for local use by default.

Safe default:

```text
localhost only
```

Do not expose the API publicly without authentication, firewall rules, HTTPS, and deployment hardening.

Optional local API token:

```powershell
$env:LOCALDOCLENS_API_TOKEN="your-token"
localdoc serve
```

Requests must include:

```text
X-LocalDoc-Token: your-token
```

Runtime security includes:

* localhost-first behavior
* optional API token
* restrictive CORS defaults
* security headers
* file name validation
* path traversal protection

---

## Verified Answer Cache

LocalDocLens can cache verified answers.

When the system confirms that an answer is directly supported by source evidence, it can reuse that answer later without repeating the full retrieval and generation process.

This improves:

* latency
* repeat question performance
* consistency
* reliability

Verified cached answers are stored locally and should not be committed.

---

## Local Learning Examples

LocalDocLens can store verified local question-answer-evidence examples.

These examples can later support:

* evaluation
* regression testing
* future fine-tuning
* local model improvement
* prompt and retrieval improvement

Export learning examples:

```powershell
localdoc export-training-data
```

Warning:

Do not publicly share exported training data if it was generated from real supplier documents.

---

## Batch Analysis

Run batch supplier analysis:

```powershell
localdoc batch-analyze
```

Run resumable batch analysis:

```powershell
localdoc batch-analyze --resume
```

The resumable pipeline can reuse existing page memory so large PDFs do not need to be fully reprocessed every time.

---

## Stress Testing

LocalDocLens includes stress testing for large supplier packets.

Generate or analyze stress test data:

```powershell
localdoc stress-test
```

Evaluate stress test results:

```powershell
localdoc stress-eval
```

Run the full benchmark suite:

```powershell
localdoc benchmark-suite
```

Benchmark outputs are saved under:

```text
artifacts
```

---

## Benchmark Results

Early benchmark results on a scanned supplier onboarding packet:

```text
Dataset:
- 1 scanned supplier onboarding packet
- 5 labeled supplier compliance questions

Answer hit accuracy: 1.0
Page/evidence hit accuracy: 1.0
Strict answer + page accuracy: 1.0
Average cold CLI answer latency: 43.5111 seconds
Minimum cold CLI answer latency: 27.2084 seconds
Maximum cold CLI answer latency: 59.5743 seconds
Average response size: 1362.4 characters
Average estimated tokens: 340.6
```

Deployment size from the early benchmark:

```text
Docs folder: 22.193 MB
LanceDB index: 0.018 MB
BM25 index: 0.027 MB
Rendered OCR pages: 0.364 MB
Project without venv: 23.049 MB
Virtual environment: 2338.167 MB
Hugging Face cache: 2467.03 MB
PaddleOCR cache: 15.645 MB
```

Warm API and verified-cache results were much faster because the server keeps model and retrieval components loaded.

Verified cached answers reached near-instant response time in local testing.

---

## Security Checks

Run the local security scanner:

```powershell
localdoc security-check
```

Run the full release check:

```powershell
localdoc release-check
```

The release check validates:

* Python compile status
* `.gitignore` protection
* required security/privacy docs
* dangerous tracked files
* local security scan
* dependency audit for the default install

For a faster local check that skips dependency audit:

```powershell
localdoc release-check --skip-pip-audit
```

Before pushing to GitHub, run:

```powershell
localdoc security-check
localdoc release-check
```

---

## Dependency Strategy

LocalDocLens uses two dependency files:

```text
requirements.txt       core/default install
requirements-ocr.txt   optional OCR install
```

The default dependency audit is run against:

```powershell
pip-audit -r requirements.txt
```

OCR dependencies are optional because OCR stacks can include heavier packages such as PaddleOCR, OpenCV, and protobuf.

This keeps the default GitHub install safer while still allowing OCR support when users need it.

---

## What Not To Commit

Do not commit:

* supplier PDFs
* scanned document images
* extracted OCR text
* SQLite databases
* LanceDB indexes
* cached answers
* learning exports
* benchmark artifacts
* agent traces
* `.env` files
* API keys
* secrets
* local model files

The `.gitignore` is configured to block these by default.

---

## Current Limitations

LocalDocLens currently focuses on text-based supplier document intelligence.

Current limitations:

* OCR can fail on extremely messy handwriting.
* Image understanding is not yet a full vision-language pipeline.
* A PDF image of a dog will not be identified as a dog unless vision ingestion is added.
* Fine-tuning is not part of the default workflow.
* Local LLM answer quality depends on the installed Ollama model.
* The system is not a replacement for legal, financial, or compliance professionals.
* Public deployment requires additional security hardening.

---

## Future Roadmap

Planned or proposed improvements:

* working-memory compression for agentic RAG
* multimodal vision ingestion for images, stamps, signatures, logos, and screenshots
* handwriting confidence scoring
* human-review queue for low-confidence fields
* stronger supplier-specific schema extraction
* asymmetric retrieval experiments with E5/BGE query-document formatting
* more benchmark datasets
* polished web UI
* local model profile options: small, medium, accurate
* quantized local model modes for CPU/GPU deployment
* optional fine-tuning after collecting enough verified examples

---

## Example End-to-End Demo

```powershell
cd C:\Users\amogh\LocalDocLens
.\.venv\Scripts\activate

localdoc ingest data/docs

localdoc facts

localdoc ask "Which suppliers have expired insurance?" --llm

localdoc report

localdoc agent-ask "Should I approve this supplier?" --file "aurora_grain_supplier_packet.pdf" --show-trace

localdoc benchmark-suite

localdoc security-check
localdoc release-check
```

---

## Summary

LocalDocLens is a local-first supplier document intelligence system that combines:

* document ingestion
* OCR
* structured supplier memory
* hybrid retrieval
* local LLM answering
* evidence citations
* answer verification
* verified caching
* local learning examples
* batch analysis
* stress testing
* agentic RAG
* security hardening

It is designed for procurement onboarding workflows where supplier decisions need to be grounded in private documents, source-page evidence, and auditable reasoning.
