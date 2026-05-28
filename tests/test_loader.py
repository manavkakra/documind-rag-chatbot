from langchain_core.documents import Document

from src.pdf_loader import PDFLoader
from src.text_splitter import DocumentChunker


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakePdfReader:
    def __init__(self, _: str) -> None:
        self.pages = [
            FakePage("Binary search works on sorted arrays."),
            FakePage("Merge sort uses divide and conquer."),
        ]


def test_pdf_loader_preserves_page_metadata(tmp_path, monkeypatch):
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr("src.pdf_loader.PdfReader", FakePdfReader)

    documents = PDFLoader().load(pdf_path)

    assert len(documents) == 2
    assert documents[0].page_content == "Binary search works on sorted arrays."
    assert documents[0].metadata["source"] == "notes.pdf"
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2


def test_chunker_creates_chunks_with_metadata():
    document = Document(
        page_content=" ".join(["dynamic programming"] * 120),
        metadata={"source": "dp.pdf", "page_number": 3},
    )
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20)

    chunks = chunker.split_documents([document])

    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "dp.pdf"
    assert chunks[0].metadata["page_number"] == 3
    assert "chunk_id" in chunks[0].metadata
    assert "chunk_index" in chunks[0].metadata
