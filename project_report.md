# Project Report: DocuMind

## 1. Abstract

DocuMind is an interactive Retrieval-Augmented Generation (RAG) chatbot designed to allow users to interact with their PDF documents using natural language. Unlike traditional LLMs that rely solely on their training data, DocuMind utilizes a hybrid retrieval approach and the Google Gemini API to generate accurate, citation-aware answers strictly grounded in the content of the uploaded documents. The system provides an end-to-end pipeline covering document extraction, chunking, dense vector embeddings, and a user-friendly conversational interface that also incorporates automated document summarization and flashcard generation.

## 2. Introduction & Motivation

As the volume of digital documents grows, manually extracting specific information from dense PDFs becomes increasingly time-consuming and error-prone. General-purpose LLMs struggle to answer questions about specific, private, or novel documents due to their lack of access to this localized context, often leading to hallucinations.

This project introduces a RAG-based solution that addresses these challenges by:
- Enabling targeted semantic search over multiple uploaded documents.
- Generating trustworthy responses by injecting relevant retrieved context into the LLM prompt.
- Fostering an interactive learning environment through automatic summarization and study flashcards.

The resulting system operates in a complete pipeline: ingesting text, indexing embeddings, retrieving context upon query, and generating grounded answers.

## 3. System Architecture

The architecture separates the data ingestion pipeline from the interactive retrieval system to ensure efficient querying and a responsive user experience.

- **Frontend Interface (Streamlit):** A custom-styled, reactive web application that orchestrates file uploads, knowledge base construction, chat history, and the display of retrieved citations.
- **Ingestion Plane (Document Processing):** Raw text is extracted via PyPDF, dynamically split into overlapping chunks, and converted into dense vector representations using Sentence Transformers (`all-MiniLM-L6-v2`).
- **Memory Plane (Vector Store):** Utilizes FAISS (Facebook AI Similarity Search) to index the chunk embeddings for rapid similarity retrieval.
- **Retrieval & Generation Core:** A Hybrid Retriever combines semantic similarity search with keyword matching to fetch the most relevant context. The retrieved chunks are formatted into a prompt and processed by the Google Gemini API to construct a response with high confidence and explicit source citations.

## 4. Implementation Details

The system integrates several modular components to facilitate a seamless RAG pipeline:

### 4.1 Document Processing & Chunking
The `DocumentChunker` (`text_splitter.py`) processes raw extracted text into manageable pieces (e.g., 1000 characters with 200 characters of overlap). This ensures context is preserved across chunk boundaries while remaining within the embedding model's optimal context window.

### 4.2 Vector Database & Hybrid Retrieval
The `FAISSVectorDatabase` handles the indexing of document vectors. Upon querying, the `HybridRetriever` (`retriever.py`) evaluates both semantic similarity (how closely the meaning aligns) and keyword frequency, scoring and ranking chunks to return the top `K` most relevant sections of the text.

### 4.3 Study Tools Pipeline
Beyond basic Q&A, the system leverages Gemini to build specialized educational features:
- **Summarizer (`summarizer.py`):** Compiles the indexed context to produce high-level structural summaries.
- **Flashcard Generator (`flashcard_generator.py`):** Identifies key definitions and concepts within the text to output formatted question-and-answer revision cards.

## 5. Testing & Evaluation

The project employs a structured test suite (`pytest`) mapped to the `tests/` directory to ensure robust pipeline behavior:

- Unit tests validate the text extraction, chunking logic, and embedding dimensions.
- The hybrid retriever is evaluated on whether the combined semantic and keyword scoring consistently ranks the most relevant chunks accurately.
- Retrieval confidence labels and source previews are surfaced directly in the UI, enabling users to independently verify the LLM's answers against the original text.

## 6. Scope & Limitations

DocuMind is an effective tool for interacting with textual PDF data but is currently bounded by the following limitations:

- **Complex Formatting:** The text extraction relies on PyPDF, which may struggle with highly complex multi-column layouts, embedded tables, or image-heavy PDFs lacking Optical Character Recognition (OCR) layers.
- **Scale Limitations:** FAISS indices are stored locally and kept in memory during the session. While highly efficient for several large documents, it is not designed to scale horizontally across enterprise-level document lakes without a dedicated external vector database backend.
- **Multi-modal Queries:** The current implementation processes and retrieves purely textual data, meaning visual charts, tables, or diagrams within the PDFs are not interpreted by the LLM.

## 7. Conclusion

DocuMind successfully demonstrates that combining a well-structured hybrid retrieval system with modern LLM capabilities creates a powerful tool for document interaction. By enforcing grounded generation (RAG) and surfacing exact citations, the system drastically reduces hallucinations and builds user trust. The addition of automated study tools further validates the transition from simple semantic search engines to comprehensive, interactive learning applications.
