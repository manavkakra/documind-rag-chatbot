from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-flash"


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str
    gemini_model_name: str
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    project_root: Path
    data_dir: Path
    vector_store_dir: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        vector_store_dir = project_root / "vector_store"

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model_name=os.getenv(
                "GEMINI_MODEL_NAME",
                DEFAULT_GEMINI_MODEL_NAME,
            ).strip(),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            ).strip(),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "5")),
            project_root=project_root,
            data_dir=data_dir,
            vector_store_dir=vector_store_dir,
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
