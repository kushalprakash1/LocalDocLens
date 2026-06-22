# LocalDocLens

LocalDocLens is a local-first supplier document intelligence system for procurement onboarding. It reads supplier PDFs, including scanned/image-only PDFs, extracts text with OCR when needed, indexes page-level evidence, and answers compliance questions using local retrieval and a local LLM.

## What It Does

LocalDocLens can answer questions like:

```powershell
localdoc ask "Which suppliers have expired insurance?" --llm
localdoc ask "Which suppliers are missing a W-9?" --llm
localdoc ask "Which documents are missing signatures?" --llm
localdoc ask "What is the bank verification status for ABC Foods LLC?" --llm
```

It returns answers with source page evidence.

## Current Features

* PDF ingestion
* Scanned PDF OCR fallback
* OCR confidence tracking
* Page-aware chunking
* E5 embeddings with `intfloat/e5-small-v2`
* LanceDB vector index
* BM25 keyword index
* Hybrid retrieval
* Local Qwen answer generation through Ollama
* Page-level evidence citations
* Benchmarking for latency, response size, index size, deployment size, answer accuracy, and page-evidence accuracy

## Working OCR Setup

The stable OCR setup is:

```text
Python 3.12
paddlepaddle==2.6.2
paddleocr==2.7.3
numpy==1.26.4
opencv-python==4.8.1.78
```

Do not upgrade PaddleOCR or PaddlePaddle without testing in a separate branch. Newer PaddleOCR/PaddlePaddle versions caused Windows CPU oneDNN runtime errors.

## Benchmark v1

Dataset:

* 1 scanned supplier onboarding packet
* 5 labeled supplier compliance questions

Results:

```text
Answer hit accuracy: 1.0
Page/evidence hit accuracy: 1.0
Strict answer + page accuracy: 1.0
Average answer latency: 43.5111s
Minimum answer latency: 27.2084s
Maximum answer latency: 59.5743s
Average response size: 1362.4 characters
Average estimated tokens: 340.6
```

Size:

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

## How To Run

Activate the environment:

```powershell
cd C:\Users\amogh\LocalDocLens
.\.venv\Scripts\activate
```

Ingest documents:

```powershell
localdoc ingest data/docs
```

Ask a question:

```powershell
localdoc ask "Which suppliers have expired insurance?" --llm
```

Run benchmark:

```powershell
python scripts\benchmark_localdoc.py --skip-ingest
```

@'

## Starting the Warm API Server

LocalDocLens includes a warm API server for low-latency supplier compliance answers.

Start the server:

```powershell
localdoc serve

## Current Limitation

The current benchmark uses cold CLI calls, so each answer starts a new process and reloads model components. This causes high latency.

## Next Step

The next optimization is warm server mode:

```text
localdoc serve
```
@'

## Supplier Compliance Report Command

LocalDocLens includes a structured supplier compliance report generator.

Run:

```powershell
localdoc report


@'

## Automatic Supplier Memory

LocalDocLens now automatically builds and refreshes supplier memory when the warm server starts.

Normal product flow:

```powershell
localdoc ingest data/docs
localdoc serve

The warm server will load the embedder, vector index, and BM25 index once, then answer questions through a FastAPI endpoint. This should reduce repeated model-loading overhead and produce a more realistic deployment latency benchmark.
