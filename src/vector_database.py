from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain_core.documents import Document

from src.embeddings import EmbeddingModel


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorSearchResult:
    document: Document
    score: float


class FAISSVectorDatabase:
    INDEX_FILE = "documind.faiss"
    DOCSTORE_FILE = "docstore.json"

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.documents: List[Document] = []

    def create_index(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("Cannot create a vector index with no documents")

        texts = [document.page_content for document in documents]
        vectors = self.embedding_model.embed_texts(texts)
        if vectors.ndim != 2:
            raise ValueError("Embedding model must return a 2D vector matrix")

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        self.index = index
        self.documents = list(documents)
        LOGGER.info("Created FAISS index with %s vectors", len(documents))

    def save_index(self, directory: str | Path) -> None:
        if self.index is None:
            raise ValueError("No FAISS index exists. Call create_index() first.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / self.INDEX_FILE))

        payload = [
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            for document in self.documents
        ]
        (path / self.DOCSTORE_FILE).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved FAISS index to %s", path)

    def load_index(self, directory: str | Path) -> None:
        path = Path(directory)
        index_file = path / self.INDEX_FILE
        docstore_file = path / self.DOCSTORE_FILE

        if not index_file.exists() or not docstore_file.exists():
            raise FileNotFoundError(f"Missing FAISS index files in {path}")

        self.index = faiss.read_index(str(index_file))
        payload = json.loads(docstore_file.read_text(encoding="utf-8"))
        self.documents = [
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
            )
            for item in payload
        ]
        LOGGER.info("Loaded FAISS index with %s documents", len(self.documents))

    def search(self, query: str, top_k: int = 5) -> List[VectorSearchResult]:
        if self.index is None:
            raise ValueError("FAISS index is not initialized")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.embedding_model.embed_query(query).astype(np.float32)
        query_matrix = np.expand_dims(query_vector, axis=0)
        limit = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_matrix, limit)

        results: List[VectorSearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            clipped_score = float(max(0.0, min(1.0, score)))
            results.append(
                VectorSearchResult(
                    document=self.documents[int(index)],
                    score=clipped_score,
                )
            )
        return results
