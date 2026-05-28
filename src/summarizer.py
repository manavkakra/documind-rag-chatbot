from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompt_templates import SUMMARY_PROMPT
from src.utils import format_citation
from src.config import DEFAULT_GEMINI_MODEL_NAME


class DocumentSummarizer:
    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL_NAME,
    ) -> None:
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for summary generation")
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.2,
        )

    def generate_summary(self, chunks: List[Document], max_chunks: int = 25) -> str:
        if not chunks:
            raise ValueError("Cannot summarize an empty document collection")

        context = self._format_context(chunks[:max_chunks])
        prompt = SUMMARY_PROMPT.format(context=context)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()

    @staticmethod
    def _format_context(chunks: List[Document]) -> str:
        return "\n\n".join(
            f"{format_citation(chunk)}\n{chunk.page_content}"
            for chunk in chunks
        )
