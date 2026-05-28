from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompt_templates import FLASHCARD_PROMPT
from src.utils import format_citation
from src.config import DEFAULT_GEMINI_MODEL_NAME


class FlashcardGenerator:
    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL_NAME,
    ) -> None:
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for flashcard generation")
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.3,
        )

    def generate_flashcards(self, chunks: List[Document], max_chunks: int = 25) -> str:
        if not chunks:
            raise ValueError("Cannot generate flashcards from an empty document collection")

        context = "\n\n".join(
            f"{format_citation(chunk)}\n{chunk.page_content}"
            for chunk in chunks[:max_chunks]
        )
        prompt = FLASHCARD_PROMPT.format(context=context)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()
