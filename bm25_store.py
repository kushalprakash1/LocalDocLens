import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from localdoc.config import BM25_INDEX_PATH


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, index_path: str | Path = BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self.bm25 = None
        self.records = []

    def build(self, records: list[dict]):
        self.records = records
        tokenized_corpus = [tokenize(r["text"]) for r in records]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "records": self.records,
                    "bm25": self.bm25,
                },
                f,
            )

    def load(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.records = data["records"]
        self.bm25 = data["bm25"]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.bm25 is None:
            self.load()

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []

        for idx, score in ranked:
            record = dict(self.records[idx])
            record["bm25_score"] = float(score)
            results.append(record)

        return results