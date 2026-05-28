# DocuMind Architecture

DocuMind is a Retrieval-Augmented Generation application for asking questions
over multiple PDFs. The system separates ingestion, embedding, retrieval, and
generation so each component can be tested and improved independently.

```text
PDF Uploads
    |
    v
PDFLoader
    |  page text + metadata
    v
DocumentChunker
    |  recursive chunks with source, page, chunk_id
    v
EmbeddingModel
    |  all-MiniLM-L6-v2 vectors
    v
FAISSVectorDatabase
    |  cosine similarity search
    v
HybridRetriever
    |  semantic score + keyword score
    v
RAGPipeline
    |  query expansion + context construction
    v
Google Gemini
    |
    v
Answer with citations
```

## Key Design Choices

- Page-level metadata is preserved before chunking so every answer can cite its
  source PDF and page number.
- Chunks use `chunk_size=1000` and `chunk_overlap=200`, which balances local
  context with retrieval precision.
- Sentence Transformer embeddings are normalized, allowing FAISS inner product
  search to behave like cosine similarity.
- Hybrid retrieval combines semantic vector search with keyword overlap for
  stronger performance on technical terms and abbreviations.
- Prompts instruct Gemini to answer only from retrieved context and to use the
  exact fallback response when evidence is missing.
