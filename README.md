# DocuMind

DocuMind is a RAG-based chatbot for interacting with PDF documents using natural language.

Upload one or more PDFs, ask questions, and get answers generated from the document content using semantic retrieval and Gemini.

The project combines:
- PDF processing
- Text embeddings
- Vector search
- Retrieval-Augmented Generation (RAG)
- Conversational AI

---

## Features

- Multi-PDF upload
- Semantic search over documents
- Citation-aware responses
- Hybrid retrieval (semantic + keyword)
- Conversational chat interface
- Document summarization
- Flashcard generation
- Retrieval confidence scoring

---

## Tech Stack

### AI / NLP
- LangChain
- Sentence Transformers
- Google Gemini API

### Vector Search
- FAISS

### Frontend
- Streamlit

### Utilities
- PyPDF
- NumPy
- Pandas
- dotenv

---

## How It Works

```text
PDF Upload
   │
   ▼
Text Extraction
   │
   ▼
Chunking
   │
   ▼
Embeddings
   │
   ▼
FAISS Vector Store
   │
   ▼
Retriever
   │
   ▼
Gemini
   │
   ▼
Answer + Citations
```

When a user asks a question:

1. Relevant chunks are retrieved from the uploaded PDFs
2. Retrieved context is sent to Gemini
3. The model generates a grounded response using only the retrieved content

---

## Installation

Clone the repository:

```bash
git clone https://github.com/manavkakra/documind-rag-chatbot.git
cd documind-rag-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_api_key

GEMINI_MODEL_NAME=gemini-2.5-flash

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

TOP_K=5
```

---

## Running the App

Start the Streamlit server:

```bash
streamlit run app.py
```

The app will open in your browser.

---

## Using the App

1. Upload one or more PDF files
2. Click **Build Knowledge Base**
3. Ask questions in the chat interface
4. View citations and retrieved sources
5. Generate summaries or flashcards if needed

---

## Example Queries

- Explain binary search and its time complexity
- Summarize the uploaded document
- What are the key differences between BFS and DFS?
- Generate flashcards for important concepts
- Which page discusses dynamic programming?

---

## Project Structure

```text
documind-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── src/
│   ├── config.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_database.py
│   ├── retriever.py
│   ├── prompt_templates.py
│   ├── rag_pipeline.py
│   ├── summarizer.py
│   ├── flashcard_generator.py
│   └── utils.py
│
├── tests/
│
├── data/
│
└── vector_store/
```

---

## Core Concepts

### Retrieval-Augmented Generation (RAG)

Instead of relying only on the LLM’s internal knowledge, the system first retrieves relevant document chunks and uses them as context for answer generation.

---

### Embeddings

Text is converted into dense vector representations using Sentence Transformers so semantically similar content can be retrieved efficiently.

---

### FAISS

FAISS is used for storing and searching embeddings using vector similarity.

---

### Hybrid Retrieval

The retriever combines:
- Semantic similarity search
- Keyword matching

This improves retrieval quality for technical and academic documents.

---

## Testing

Run tests with:

```bash
pytest
```