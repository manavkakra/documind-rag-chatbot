from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower() or ".pdf"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{safe_stem or 'document'}{suffix}"


def save_uploaded_file(uploaded_file: Any, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / safe_filename(uploaded_file.name)
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def format_citation(document: Document) -> str:
    source = document.metadata.get("source", "Unknown document")
    page = document.metadata.get("page_number", "Unknown")
    return f"[Source: {source}, Page {page}]"


def confidence_label(score: float) -> str:
    if score >= 0.72:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"
