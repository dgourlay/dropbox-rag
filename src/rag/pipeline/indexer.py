from __future__ import annotations

from typing import TYPE_CHECKING

from rag.types import (
    EmbeddedChunk,
    FileType,
    QdrantPayloadModel,
    RecordType,
    VectorPoint,
)

if TYPE_CHECKING:
    from rag.protocols import VectorStore


class QdrantIndexer:
    """Constructs VectorPoints from embedded chunks and upserts to Qdrant."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def index_document(
        self,
        doc_id: str,
        title: str,
        file_path: str,
        folder_path: str,
        folder_ancestors: list[str],
        file_type: FileType,
        modified_at: str,
        embedded_chunks: list[EmbeddedChunk],
    ) -> int:
        """Index embedded chunks into Qdrant. Returns count of points upserted."""
        points: list[VectorPoint] = []

        for ec in embedded_chunks:
            chunk = ec.chunk
            payload = QdrantPayloadModel(
                record_type=RecordType.CHUNK,
                doc_id=doc_id,
                section_id=chunk.section_id,
                chunk_id=chunk.chunk_id,
                title=title,
                file_path=file_path,
                folder_path=folder_path,
                folder_ancestors=folder_ancestors,
                file_type=file_type,
                modified_at=modified_at,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_heading=chunk.section_heading,
                chunk_order=chunk.chunk_order,
                token_count=chunk.token_count,
                citation_label=chunk.citation_label,
                text=chunk.text,
            )
            points.append(
                VectorPoint(
                    point_id=chunk.chunk_id,
                    vector=ec.vector,
                    payload=payload,
                )
            )

        if points:
            self._vector_store.upsert_points(doc_id, points)
            keep_ids = {p.point_id for p in points}
            self._vector_store.delete_stale_points(doc_id, keep_ids)

        return len(points)
