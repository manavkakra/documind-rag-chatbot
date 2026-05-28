from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from src.retriever import HybridRetriever, RetrievedChunk
from src.utils import format_citation
from src.config import DEFAULT_GEMINI_MODEL_NAME


LOGGER = logging.getLogger(__name__)
NOT_FOUND_RESPONSE = "I could not find this information in the uploaded documents."


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    expanded_question: str
    retrieved_chunks: List[RetrievedChunk]


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        gemini_api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL_NAME,
    ) -> None:
        self.retriever = retriever
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._llm: ChatGoogleGenerativeAI | None = None

    def answer_question(
        self,
        question: str,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> RAGResponse:
        question = question.strip()
        if not question:
            return RAGResponse(
                answer="Please enter a question.",
                expanded_question="",
                retrieved_chunks=[],
            )

        expanded_question = self.expand_query(question)
        retrieved_chunks = self.retriever.retrieve(expanded_question)

        if not retrieved_chunks:
            return RAGResponse(
                answer=NOT_FOUND_RESPONSE,
                expanded_question=expanded_question,
                retrieved_chunks=[],
            )

        context = self._build_context(retrieved_chunks)
        history_text = self._format_history(chat_history or [])
        user_prompt = RAG_USER_PROMPT.format(
            chat_history=history_text,
            expanded_question=expanded_question,
            context=context,
            question=question,
        )

        response = self._get_llm().invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = str(response.content).strip()
        if not answer:
            answer = NOT_FOUND_RESPONSE
        elif answer != NOT_FOUND_RESPONSE:
            answer = self._ensure_citations(answer, retrieved_chunks)

        return RAGResponse(
            answer=answer,
            expanded_question=expanded_question,
            retrieved_chunks=retrieved_chunks,
        )

    @staticmethod
    def expand_query(question: str) -> str:
        cleaned = " ".join(question.strip().split())
        word_count = len(cleaned.split())
        if word_count <= 4 and not cleaned.endswith("?"):
            return (
                f"Explain {cleaned}, including definition, working principle, "
                "important steps, examples, and time complexity if applicable."
            )
        return cleaned

    @staticmethod
    def _build_context(chunks: List[RetrievedChunk]) -> str:
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            citation = format_citation(chunk.document)
            context_blocks.append(
                f"Context {index} {citation}\n"
                f"{chunk.document.page_content}"
            )
        return "\n\n".join(context_blocks)

    @staticmethod
    def _format_history(chat_history: List[Dict[str, str]], limit: int = 6) -> str:
        if not chat_history:
            return "No previous conversation."
        recent_messages = chat_history[-limit:]
        lines = [
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in recent_messages
        ]
        return "\n".join(lines)

    @staticmethod
    def _ensure_citations(answer: str, chunks: List[RetrievedChunk]) -> str:
        if "[Source:" in answer:
            return answer

        citations = []
        for chunk in chunks:
            citation = format_citation(chunk.document)
            if citation not in citations:
                citations.append(citation)

        if not citations:
            return answer
        return f"{answer}\n\nSources: {' '.join(citations[:3])}"

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini answer generation")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.gemini_api_key,
            temperature=0.1,
        )
