from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        LOGGER.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self._model.encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
