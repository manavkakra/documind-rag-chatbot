from __future__ import annotations

import logging
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import stable_hash


LOGGER = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for document in documents:
            split_texts = self._splitter.split_text(document.page_content)
            for chunk_index, text in enumerate(split_texts):
                chunk_id = stable_hash(
                    "|".join(
                        [
                            str(document.metadata.get("source", "")),
                            str(document.metadata.get("page_number", "")),
                            str(chunk_index),
                            text,
                        ]
                    )
                )
                metadata = {
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                }
                chunks.append(Document(page_content=text, metadata=metadata))

        LOGGER.info("Created %s chunks from documents", len(chunks))
        return chunks
