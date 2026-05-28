from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from pypdf import PdfReader

from src.utils import clean_text


LOGGER = logging.getLogger(__name__)


class PDFLoader:
    def load(self, file_path: str | Path) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        LOGGER.info("Loading PDF: %s", path)
        reader = PdfReader(str(path))
        documents: List[Document] = []

        for page_index, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            text = clean_text(raw_text)
            if not text:
                LOGGER.debug("Skipping empty page %s in %s", page_index + 1, path.name)
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "page_number": page_index + 1,
                        "page_index": page_index,
                    },
                )
            )

        if not documents:
            raise ValueError(f"No extractable text found in PDF: {path.name}")

        return documents

    def load_many(self, file_paths: Iterable[str | Path]) -> List[Document]:
        all_documents: List[Document] = []
        for file_path in file_paths:
            all_documents.extend(self.load(file_path))
        return all_documents
