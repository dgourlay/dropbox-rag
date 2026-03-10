from __future__ import annotations

from unittest.mock import MagicMock

from rag.pipeline.indexer import QdrantIndexer
from rag.types import Chunk, EmbeddedChunk, FileType, RecordType


def _make_embedded_chunk(
    doc_id: str = "doc-1",
    chunk_order: int = 0,
    chunk_id: str | None = None,
) -> EmbeddedChunk:
    cid = chunk_id or f"chunk-{chunk_order}"
    chunk = Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        section_id="sec-1",
        chunk_order=chunk_order,
        text=f"Text for chunk {chunk_order}",
        text_normalized=f"text for chunk {chunk_order}",
        page_start=1,
        page_end=2,
        section_heading="Introduction",
        citation_label=f"[{chunk_order + 1}]",
        token_count=42,
    )
    return EmbeddedChunk(chunk=chunk, vector=[0.1] * 1024)


class TestQdrantIndexer:
    def test_index_document_constructs_correct_payload(self) -> None:
        mock_store = MagicMock()
        indexer = QdrantIndexer(mock_store)
        ec = _make_embedded_chunk(doc_id="doc-1", chunk_order=0)

        count = indexer.index_document(
            doc_id="doc-1",
            title="Test Doc",
            file_path="/docs/test.pdf",
            folder_path="/docs",
            folder_ancestors=["/"],
            file_type=FileType.PDF,
            modified_at="2026-01-01T00:00:00Z",
            embedded_chunks=[ec],
        )

        assert count == 1
        upserted_points = mock_store.upsert_points.call_args[0][1]
        point = upserted_points[0]
        assert point.point_id == "chunk-0"
        assert point.vector == [0.1] * 1024
        payload = point.payload
        assert payload.record_type == RecordType.CHUNK
        assert payload.doc_id == "doc-1"
        assert payload.title == "Test Doc"
        assert payload.file_path == "/docs/test.pdf"
        assert payload.folder_path == "/docs"
        assert payload.folder_ancestors == ["/"]
        assert payload.file_type == FileType.PDF
        assert payload.modified_at == "2026-01-01T00:00:00Z"
        assert payload.page_start == 1
        assert payload.page_end == 2
        assert payload.section_heading == "Introduction"
        assert payload.chunk_order == 0
        assert payload.token_count == 42
        assert payload.citation_label == "[1]"
        assert payload.text == "Text for chunk 0"

    def test_stale_point_deletion_called_with_correct_ids(self) -> None:
        mock_store = MagicMock()
        indexer = QdrantIndexer(mock_store)
        chunks = [_make_embedded_chunk(chunk_order=i) for i in range(3)]

        indexer.index_document(
            doc_id="doc-1",
            title="Doc",
            file_path="/test.txt",
            folder_path="/",
            folder_ancestors=[],
            file_type=FileType.TXT,
            modified_at="2026-01-01T00:00:00Z",
            embedded_chunks=chunks,
        )

        mock_store.delete_stale_points.assert_called_once()
        call_args = mock_store.delete_stale_points.call_args[0]
        assert call_args[0] == "doc-1"
        assert call_args[1] == {"chunk-0", "chunk-1", "chunk-2"}

    def test_empty_chunks_no_upsert(self) -> None:
        mock_store = MagicMock()
        indexer = QdrantIndexer(mock_store)

        count = indexer.index_document(
            doc_id="doc-1",
            title="Empty",
            file_path="/empty.md",
            folder_path="/",
            folder_ancestors=[],
            file_type=FileType.MD,
            modified_at="2026-01-01T00:00:00Z",
            embedded_chunks=[],
        )

        assert count == 0
        mock_store.upsert_points.assert_not_called()
        mock_store.delete_stale_points.assert_not_called()

    def test_multiple_chunks_all_have_correct_doc_id(self) -> None:
        mock_store = MagicMock()
        indexer = QdrantIndexer(mock_store)
        chunks = [_make_embedded_chunk(doc_id="doc-99", chunk_order=i) for i in range(5)]

        count = indexer.index_document(
            doc_id="doc-99",
            title="Multi",
            file_path="/multi.docx",
            folder_path="/reports",
            folder_ancestors=["/"],
            file_type=FileType.DOCX,
            modified_at="2026-06-15T12:00:00Z",
            embedded_chunks=chunks,
        )

        assert count == 5
        upserted_points = mock_store.upsert_points.call_args[0][1]
        for point in upserted_points:
            assert point.payload.doc_id == "doc-99"
            assert point.payload.file_path == "/multi.docx"
            assert point.payload.folder_path == "/reports"
            assert point.payload.file_type == FileType.DOCX

    def test_returns_correct_count(self) -> None:
        mock_store = MagicMock()
        indexer = QdrantIndexer(mock_store)
        chunks = [_make_embedded_chunk(chunk_order=i) for i in range(7)]

        count = indexer.index_document(
            doc_id="doc-1",
            title="Count",
            file_path="/count.txt",
            folder_path="/",
            folder_ancestors=[],
            file_type=FileType.TXT,
            modified_at="2026-01-01T00:00:00Z",
            embedded_chunks=chunks,
        )

        assert count == 7

    def test_upsert_called_with_doc_id(self) -> None:
        mock_store = MagicMock()
        indexer = QdrantIndexer(mock_store)
        chunks = [_make_embedded_chunk()]

        indexer.index_document(
            doc_id="doc-abc",
            title="T",
            file_path="/t.txt",
            folder_path="/",
            folder_ancestors=[],
            file_type=FileType.TXT,
            modified_at="2026-01-01T00:00:00Z",
            embedded_chunks=chunks,
        )

        assert mock_store.upsert_points.call_args[0][0] == "doc-abc"
