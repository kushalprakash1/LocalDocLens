from sentence_transformers import SentenceTransformer
from localdoc.config import EMBEDDING_MODEL_NAME


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        formatted = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(
            formatted,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        formatted = f"query: {query}"
        vector = self.model.encode(
            [formatted],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()