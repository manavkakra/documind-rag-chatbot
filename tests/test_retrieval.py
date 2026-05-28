import numpy as np
from langchain_core.documents import Document

from src.retriever import HybridRetriever
from src.vector_database import FAISSVectorDatabase


class FakeEmbeddingModel:
    dimension = 4

    vocabulary = {
        "binary": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "search": np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32),
        "graph": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "traversal": np.array([0.0, 0.8, 0.2, 0.0], dtype=np.float32),
        "database": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    def embed_texts(self, texts):
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, query):
        return self._embed(query).astype(np.float32)

    def _embed(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            vector += self.vocabulary.get(token.strip(".,?"), 0.01)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.ones(self.dimension, dtype=np.float32) / 2.0
        return vector / norm


def make_documents():
    return [
        Document(
            page_content="binary search finds an item in a sorted array",
            metadata={"source": "algorithms.pdf", "page_number": 10, "chunk_id": "a"},
        ),
        Document(
            page_content="graph traversal includes breadth first search",
            metadata={"source": "graphs.pdf", "page_number": 4, "chunk_id": "b"},
        ),
        Document(
            page_content="database indexes speed up lookup queries",
            metadata={"source": "db.pdf", "page_number": 2, "chunk_id": "c"},
        ),
    ]


def test_faiss_vector_database_search_returns_ranked_documents():
    vector_db = FAISSVectorDatabase(embedding_model=FakeEmbeddingModel())
    vector_db.create_index(make_documents())

    results = vector_db.search("binary search", top_k=2)

    assert len(results) == 2
    assert results[0].document.metadata["source"] == "algorithms.pdf"
    assert results[0].score >= results[1].score


def test_hybrid_retriever_returns_scores_and_metadata():
    vector_db = FAISSVectorDatabase(embedding_model=FakeEmbeddingModel())
    vector_db.create_index(make_documents())
    retriever = HybridRetriever(vector_db=vector_db, top_k=2)

    results = retriever.retrieve("graph traversal")

    assert len(results) == 2
    assert results[0].document.metadata["source"] == "graphs.pdf"
    assert results[0].combined_score > 0
    assert results[0].semantic_score >= 0
    assert results[0].keyword_score >= 0
