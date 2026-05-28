from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document

from src.vector_database import FAISSVectorDatabase


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    semantic_score: float
    keyword_score: float
    combined_score: float


class HybridRetriever:
    def __init__(
        self,
        vector_db: FAISSVectorDatabase,
        top_k: int = 5,
        semantic_weight: float = 0.75,
        keyword_weight: float = 0.25,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("weights must be non-negative")
        if semantic_weight + keyword_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")

        self.vector_db = vector_db
        self.top_k = top_k
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        semantic_results = self.vector_db.search(query, top_k=max(self.top_k * 3, self.top_k))
        semantic_by_id: Dict[str, float] = {
            self._document_id(result.document): result.score for result in semantic_results
        }

        query_terms = self._tokenize(query)
        candidates = {self._document_id(result.document): result.document for result in semantic_results}

        keyword_scores: Dict[str, float] = {}
        for document in self.vector_db.documents:
            document_id = self._document_id(document)
            score = self._keyword_score(query_terms, document.page_content)
            if score > 0:
                candidates[document_id] = document
                keyword_scores[document_id] = score

        ranked: List[RetrievedChunk] = []
        for document_id, document in candidates.items():
            semantic_score = semantic_by_id.get(document_id, 0.0)
            keyword_score = keyword_scores.get(
                document_id,
                self._keyword_score(query_terms, document.page_content),
            )
            combined_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_score
            )
            ranked.append(
                RetrievedChunk(
                    document=document,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                )
            )

        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        LOGGER.info("Retrieved %s chunks for query: %s", min(len(ranked), self.top_k), query)
        return ranked[: self.top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(token) > 2
        }

    @classmethod
    def _keyword_score(cls, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = cls._tokenize(text)
        overlap = query_terms.intersection(text_terms)
        return len(overlap) / len(query_terms)

    @staticmethod
    def _document_id(document: Document) -> str:
        return str(document.metadata.get("chunk_id") or id(document))
