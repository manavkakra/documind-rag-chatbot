# Project Report: DocuMind

## 1. Project Overview
**DocuMind** is a sophisticated, interactive Retrieval-Augmented Generation (RAG) chatbot designed to enable users to "converse" with their PDF documents using natural language. By uploading one or more PDFs, users can ask questions and receive accurate, citation-aware answers generated entirely from the content of their uploaded documents.

The core motivation behind DocuMind is to simplify information retrieval from dense documents, making it easier for users to study, research, and extract knowledge without needing to manually read through pages of text.

## 2. Key Features
- **Multi-PDF Upload:** Users can upload multiple PDF documents simultaneously to build a comprehensive knowledge base.
- **Conversational Chat Interface:** A user-friendly chat UI built with Streamlit for asking natural language questions.
- **Citation-Aware Responses:** Answers are grounded in the uploaded documents and include citations to the exact sources and pages.
- **Hybrid Retrieval System:** Combines semantic similarity search with keyword matching for highly accurate context retrieval, especially useful for technical and academic texts.
- **Retrieval Confidence Scoring:** Displays semantic and keyword scores, providing transparency into the reliability of the retrieved context.
- **Study Tools:**
  - **Document Summarization:** Automatically generates comprehensive summaries of the uploaded knowledge base.
  - **Flashcard Generation:** Automatically generates question-and-answer revision flashcards from the text to aid in studying.

## 3. Technology Stack
The application is built using modern AI and web development libraries:
- **Frontend / UI:** [Streamlit](https://streamlit.io/) for a reactive, pure-Python web interface.
- **LLM / AI:** [Google Gemini API](https://ai.google.dev/) for generating grounded answers, summaries, and flashcards.
- **Vector Search & Embeddings:** 
  - **FAISS** (Facebook AI Similarity Search) for efficient storage and similarity search of document vectors.
  - **Sentence Transformers** (`all-MiniLM-L6-v2`) for generating dense vector representations of text.
- **Orchestration & NLP Utilities:**
  - **LangChain** concepts applied for structuring the pipeline.
  - **PyPDF** for extracting text from PDF files.
- **Data Handling:** NumPy, Pandas for numerical and tabular operations.

## 4. System Architecture
DocuMind utilizes a standard RAG architecture enhanced with hybrid retrieval and study tools.

### 4.1. Ingestion Pipeline
1. **Document Loading (`pdf_loader.py`):** Extracts raw text from uploaded PDF files.
2. **Text Splitting / Chunking (`text_splitter.py`):** Splits the raw text into manageable chunks (e.g., 1000 characters with 200 character overlap) to preserve context while ensuring chunks fit into the embedding model's context window.
3. **Embedding (`embeddings.py`):** Converts the text chunks into dense mathematical vectors using Sentence Transformers.
4. **Vector Database (`vector_database.py`):** Indexes and stores the chunks and their vectors using FAISS.

### 4.2. Retrieval & Generation Pipeline
1. **User Query:** The user enters a question in the Streamlit interface.
2. **Hybrid Retrieval (`retriever.py`):** The query is embedded and compared against the FAISS index. The retriever uses a hybrid approach (Semantic + Keyword scoring) to fetch the top `K` most relevant chunks.
3. **Prompt Construction (`prompt_templates.py` & `rag_pipeline.py`):** The retrieved chunks are formatted into a context block and combined with the user query into a strict prompt instructing the LLM to only use the provided context.
4. **LLM Generation:** The Google Gemini model generates a detailed answer along with citations referring to the original documents.

### 4.3. Study Tools Generation
- **Summarizer (`summarizer.py`):** Passes the entire indexed context to Gemini to extract the main ideas and summarize the document.
- **Flashcard Generator (`flashcard_generator.py`):** Instructs the LLM to identify key definitions and concepts to create structured Q&A cards.

## 5. UI/UX Design
The application features a custom-styled Streamlit interface tailored for readability and a premium feel:
- **Dark Mode Aesthetic:** Custom CSS defines a sleek, modern dark theme with distinct branding (`var(--dm-bg)`, `var(--dm-surface)`).
- **Responsive Layout:** A side panel manages the Knowledge Base build process and study tools, while the main window handles the interactive chat.
- **Source Cards:** Retrieved contexts are displayed in expandable UI cards showing the confidence score and an excerpt of the text, ensuring full transparency in the generation process.

## 6. Setup and Installation
The project requires Python and can be set up in a virtual environment:
1. Clone the repository and install dependencies (`pip install -r requirements.txt`).
2. Configure `.env` with a `GEMINI_API_KEY`.
3. Launch the application using `streamlit run app.py`.

## 7. Conclusion
DocuMind successfully demonstrates how Large Language Models and Vector Databases can be integrated to create powerful document interaction tools. By emphasizing grounded generation (RAG) and hybrid retrieval, it minimizes hallucinations and ensures that the information provided to the user is accurate, verifiable, and highly relevant to their own documents.
