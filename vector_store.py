import lancedb
import pandas as pd
from localdoc.config import LANCEDB_PATH


TABLE_NAME = "chunks"


class VectorStore:
    def __init__(self, db_path: str = LANCEDB_PATH):
        self.db = lancedb.connect(db_path)

    def reset_table(self):
        try:
            self.db.drop_table(TABLE_NAME)
        except Exception:
            pass

    def add_chunks(self, chunks: list[dict]):
        if not chunks:
            return

        df = pd.DataFrame(chunks)

        try:
            table = self.db.open_table(TABLE_NAME)
            table.add(df)
        except Exception:
            self.db.create_table(TABLE_NAME, data=df)

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        table = self.db.open_table(TABLE_NAME)
        results = table.search(query_vector).limit(top_k).to_list()
        return results