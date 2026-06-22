# LocalDocLens Project Notes

## Project Summary

LocalDocLens is a local-first supplier document intelligence system for procurement onboarding. It reads supplier PDFs, including scanned/image-only PDFs, extracts text with OCR when needed, indexes document chunks with hybrid retrieval, and answers supplier compliance questions using a local LLM with page-level evidence.

The goal is to help procurement or supplier onboarding teams quickly identify missing or risky supplier documentation such as expired insurance, missing W-9 forms, missing signatures, and bank verification issues.

## Current Working Status

The MVP is working end-to-end.

Completed capabilities:

* Reads supplier PDFs from `data/docs`
* Supports selectable-text PDFs
* Supports scanned/image-only PDFs using OCR fallback
* Uses PaddleOCR for OCR extraction
* Tracks OCR confidence per chunk
* Chunks documents by page
* Embeds chunks with `intfloat/e5-small-v2`
* Stores dense vectors in LanceDB
* Builds BM25 keyword index
* Uses hybrid retrieval with dense search + BM25
* Uses local Qwen through Ollama for evidence-backed answers
* Returns source page evidence
* Benchmarks latency, response size, index size, deployment size, answer accuracy, and page-evidence accuracy

## Working Environment

The working OCR/runtime combo is:

* Python: 3.12
* PaddlePaddle: `2.6.2`
* PaddleOCR: `2.7.3`
* NumPy: `1.26.4`
* OpenCV: `4.8.1.78`
* Embedding model: `intfloat/e5-small-v2`
* Vector DB: LanceDB
* Keyword retrieval: BM25 with `rank-bm25`
* Local LLM: Qwen through Ollama

Important note: PaddleOCR 3.x and PaddlePaddle 3.x caused Windows CPU oneDNN/PIR runtime errors. The stable fix was downgrading to PaddlePaddle 2.6.2 with PaddleOCR 2.7.3.

## OCR Debugging History

Initial PaddleOCR setup failed because the virtual environment had Python 3.14 compiled `.pyd` files inside a Python 3.12 environment.

The environment was fixed by deleting and rebuilding `.venv` cleanly.

Later, PaddleOCR 3.x crashed with oneDNN/PIR errors:

* `ConvertPirAttribute2RuntimeAttribute not support`
* `OneDnnContext does not have the input Filter`
* `fused_conv2d` runtime errors

Downgrading to this version combo fixed OCR:

* `paddlepaddle==2.6.2`
* `paddleocr==2.7.3`
* `numpy==1.26.4`
* `opencv-python==4.8.1.78`

OCR now works on scanned PDFs and creates indexed chunks with OCR confidence around `0.986–0.991`.

## Test Dataset

Current test file:

* `data/docs/scanned_supplier_packet.pdf`

This is an image-only scanned supplier packet generated from the sample supplier packet. It forces OCR because selectable PDF text extraction returns little or no text.

Indexed OCR pages:

* Page 1: supplier summary, missing W-9 note, pending supplier agreement note
* Page 2: certificate of insurance, expired policy date `03/12/2025`
* Page 3: bank verification and ACH details
* Page 4: supplier agreement signature page, missing supplier signature

## Benchmark v1 Results

Benchmark file:

* `artifacts/benchmark_report.json`
* `artifacts/benchmark_answers.csv`

Dataset:

* 1 scanned supplier onboarding packet
* 5 labeled compliance questions

Questions tested:

1. Which suppliers have expired insurance?
2. Which suppliers are missing a W-9?
3. Which documents are missing signatures?
4. What is the bank verification status for ABC Foods LLC?
5. Summarize ABC Foods LLC's compliance status.

Results:

* Number of questions: 5
* Answer hit accuracy: `1.0`
* Page/evidence hit accuracy: `1.0`
* Strict answer + page accuracy: `1.0`
* Average answer latency: `43.5111s`
* Minimum answer latency: `27.2084s`
* Maximum answer latency: `59.5743s`
* Average response size: `1362.4` characters
* Average estimated tokens: `340.6`

Size measurements:

* Docs folder: `22.193 MB`
* LanceDB folder: `0.018 MB`
* BM25 index folder: `0.027 MB`
* Rendered OCR pages folder: `0.364 MB`
* Artifacts folder: `0.011 MB`
* Virtual environment size: `2338.167 MB`
* Project without venv: `23.049 MB`
* Hugging Face cache: `2467.03 MB`
* PaddleOCR cache: `15.645 MB`

## Benchmark Interpretation

Accuracy is currently strong on the labeled test packet:

* The system answered all 5 supplier compliance questions correctly.
* The system cited or retrieved the correct expected evidence pages for all 5 questions.
* OCR extraction worked correctly on the scanned PDF.

The main weakness is latency.

Current average answer latency is about `43.5s`, which is too slow for a production demo. The cause is mostly cold-start CLI behavior: each `localdoc ask` command starts a new Python process and reloads model components.

## Latency Bottleneck

The current CLI benchmark is slow because every question runs as a new process.

Cold CLI path:

1. Start Python process
2. Import project
3. Load embedding model
4. Open LanceDB/BM25
5. Retrieve chunks
6. Call local Qwen through Ollama
7. Generate answer
8. Exit process

This causes repeated model loading overhead.

The next optimization is warm server mode.

## Next Optimization Plan

Next feature: `localdoc serve`

Goal:

Start LocalDocLens once, load models once, and answer multiple questions through an API.

Expected architecture:

* FastAPI server
* Load E5 embedder once at startup
* Load/open LanceDB once at startup
* Load BM25 index once at startup
* Keep retrieval pipeline warm
* Send prompts to Ollama without restarting Python
* Add `/ask` endpoint
* Add `/health` endpoint
* Add warm latency benchmark script

Expected improvement:

* Reduce repeated CLI startup overhead
* Measure warm answer latency separately from cold CLI latency
* Make the system closer to real deployment architecture

## Next Commands to Demo Current System

Ingest documents:

```powershell
localdoc ingest data/docs
```

Ask compliance questions:

```powershell
localdoc ask "Which suppliers have expired insurance?" --llm
localdoc ask "Which suppliers are missing a W-9?" --llm
localdoc ask "Which documents are missing signatures?" --llm
localdoc ask "What is the bank verification status for ABC Foods LLC?" --llm
localdoc ask "Summarize ABC Foods LLC's compliance status." --llm
```

Run benchmark:

```powershell
python scripts\benchmark_localdoc.py --skip-ingest
```

Inspect indexed OCR chunks:

```powershell
python -c "import lancedb; db=lancedb.connect('data/db/lancedb'); print(db.list_tables()); t=db.open_table('chunks'); df=t.to_pandas(); print(df[['chunk_id','page_number','extraction_method','ocr_confidence','text']].to_string(max_colwidth=300))"
```
## Known Limitations

* Benchmark dataset currently has only one supplier packet.
* Accuracy needs to be tested on more supplier packets.
* Latency is high in CLI mode.
* Current benchmark measures cold process latency.
* Deployment size is large because of local ML dependencies and model caches.
* OCR confidence is tracked, but there is not yet a visual bounding-box viewer.
* No UI yet.
* No automatic structured risk report command yet.

## Next Milestones

1. Add `localdoc serve` warm FastAPI server
2. Add warm API benchmark
3. Add `localdoc report` structured supplier risk report
4. Add more synthetic supplier packets
5. Add retrieval-only benchmark
6. Add CSV/JSON report export
7. Add README demo screenshots
8. Push to GitHub
