import numpy as np

from src.embeddings import EmbeddingModel


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []
        for text in texts:
            base = float(len(text))
            vector = np.array([base, base + 1.0, base + 2.0], dtype=np.float32)
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_embedding_model_returns_normalized_vectors(monkeypatch):
    monkeypatch.setattr("src.embeddings.SentenceTransformer", FakeSentenceTransformer)
    model = EmbeddingModel()

    vectors = model.embed_texts(["graph algorithms", "binary search"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_query_rejects_empty_query(monkeypatch):
    monkeypatch.setattr("src.embeddings.SentenceTransformer", FakeSentenceTransformer)
    model = EmbeddingModel()

    try:
        model.embed_query("   ")
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query")
