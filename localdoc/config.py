from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_DIR = DATA_DIR / "db"
INDEX_DIR = DATA_DIR / "indexes"
RENDERED_PAGES_DIR = DATA_DIR / "rendered_pages"
MIN_TEXT_CHARS_BEFORE_OCR = 30

LANCEDB_PATH = str(DB_DIR / "lancedb")
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"

CHUNK_MAX_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

TOP_K_VECTOR = 12
TOP_K_BM25 = 12
TOP_K_FINAL = 8
