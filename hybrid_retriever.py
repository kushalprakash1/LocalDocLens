from localdoc.config import TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL
from localdoc.indexing.embedder import Embedder
from localdoc.indexing.vector_store import VectorStore
from localdoc.indexing.bm25_store import BM25Store


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Combines BM25 and vector results.
    RRF gives points based on rank, not raw scores.
    """
    scores = {}
    merged = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            chunk_id = item["chunk_id"]
            merged[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    final = []

    for cid in ranked_ids:
        item = merged[cid]
        item["hybrid_score"] = scores[cid]
        final.append(item)

    return final


class HybridRetriever:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

    def search(self, query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
        query_vector = self.embedder.embed_query(query)

        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=TOP_K_VECTOR,
        )

        bm25_results = self.bm25_store.search(
            query=query,
            top_k=TOP_K_BM25,
        )

        fused = reciprocal_rank_fusion([vector_results, bm25_results])

        return fused[:top_k]