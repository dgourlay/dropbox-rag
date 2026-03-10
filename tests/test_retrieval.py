from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from rag.retrieval.engine import RRF_K, RetrievalEngine, rrf_fuse
from rag.retrieval.query_analyzer import analyze_query
from rag.types import (
    Citation,
    CitedEvidence,
    RecordType,
    RetrievalResult,
    SearchFilters,
    SearchHit,
)

# --- Helpers ---


def _make_hit(
    point_id: str,
    score: float = 0.9,
    doc_id: str = "doc1",
    text: str = "some text",
) -> SearchHit:
    return SearchHit(
        point_id=point_id,
        score=score,
        record_type=RecordType.CHUNK,
        doc_id=doc_id,
        text=text,
        payload={"file_path": "/docs/test.pdf", "title": "Test", "modified_at": "2025-01-01"},
    )


def _make_cited(hit: SearchHit) -> CitedEvidence:
    return CitedEvidence(
        text=hit.text,
        citation=Citation(
            title="Test",
            path="/docs/test.pdf",
            section=None,
            pages=None,
            modified="2025-01-01",
            label="test.pdf",
        ),
        score=hit.score,
        record_type=hit.record_type.value,
    )


# --- RRF Fusion Tests ---


class TestRRFFuse:
    def test_single_source_scores(self) -> None:
        """Dense-only hits get correct RRF scores."""
        hits = [_make_hit("a"), _make_hit("b")]
        fused = rrf_fuse(hits, [])
        assert len(fused) == 2
        assert fused[0].point_id == "a"
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))
        assert fused[1].score == pytest.approx(1.0 / (RRF_K + 2))

    def test_both_sources_combined_score(self) -> None:
        """Hit appearing in both lists gets summed RRF score."""
        dense = [_make_hit("a"), _make_hit("b")]
        keyword = [_make_hit("b"), _make_hit("c")]
        fused = rrf_fuse(dense, keyword)

        scores = {h.point_id: h.score for h in fused}
        # "b" appears at rank 1 in dense (1/(61+1)) and rank 0 in keyword (1/(60+1))
        expected_b = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
        assert scores["b"] == pytest.approx(expected_b)
        assert fused[0].point_id == "b"  # highest combined score

    def test_deduplication(self) -> None:
        """Same point_id in both lists produces single entry."""
        dense = [_make_hit("x")]
        keyword = [_make_hit("x")]
        fused = rrf_fuse(dense, keyword)
        assert len(fused) == 1

    def test_empty_inputs(self) -> None:
        """Empty inputs return empty list."""
        assert rrf_fuse([], []) == []

    def test_ordering_by_score(self) -> None:
        """Results are sorted descending by RRF score."""
        dense = [_make_hit("a"), _make_hit("b"), _make_hit("c")]
        keyword = [_make_hit("c"), _make_hit("a")]
        fused = rrf_fuse(dense, keyword)
        scores = [h.score for h in fused]
        assert scores == sorted(scores, reverse=True)


# --- Query Analyzer Tests ---


class TestQueryAnalyzer:
    def test_short_query_is_broad(self) -> None:
        result = analyze_query("machine learning")
        assert result.classification == "broad"

    def test_question_is_broad(self) -> None:
        result = analyze_query("what are the main findings of the research paper")
        assert result.classification == "broad"

    def test_specific_query(self) -> None:
        result = analyze_query(
            "implementation details of the gradient descent optimizer in section 3.2"
        )
        assert result.classification == "specific"

    def test_folder_hint_extraction(self) -> None:
        result = analyze_query("find documents in /docs/reports")
        assert result.folder_hint == "/docs/reports"

    def test_date_hint_extraction(self) -> None:
        result = analyze_query("changes since 2025-01-15")
        assert result.date_hint == "2025-01-15"

    def test_no_hints(self) -> None:
        result = analyze_query("how does the system work")
        assert result.folder_hint is None
        assert result.date_hint is None

    def test_frozen_dataclass(self) -> None:
        result = analyze_query("test")
        with pytest.raises(AttributeError):
            result.classification = "specific"  # type: ignore[misc]


# --- RetrievalEngine Tests ---


class TestRetrievalEngine:
    def _build_engine(self) -> tuple[RetrievalEngine, MagicMock, MagicMock, MagicMock, MagicMock]:
        vector_store = MagicMock()
        embedder = MagicMock()
        reranker = MagicMock()
        citation_assembler = MagicMock()

        engine = RetrievalEngine(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            citation_assembler=citation_assembler,
            top_k_candidates=30,
            top_k_final=10,
        )
        return engine, vector_store, embedder, reranker, citation_assembler

    def test_search_pipeline_order(self) -> None:
        """search() calls embed -> dense -> keyword -> rerank -> citations."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        query_vec = [0.1] * 1024
        embedder.embed_query.return_value = query_vec

        dense_hits = [_make_hit("d1"), _make_hit("d2")]
        keyword_hits = [_make_hit("k1")]
        vs.query_dense.return_value = dense_hits
        vs.query_keyword.return_value = keyword_hits

        reranked = [_make_hit("d1")]
        reranker.rerank.return_value = reranked

        cited = [_make_cited(reranked[0])]
        citations.assemble_citations.return_value = cited

        result = engine.search("test query")

        embedder.embed_query.assert_called_once_with("test query")
        vs.query_dense.assert_called_once()
        vs.query_keyword.assert_called_once()
        reranker.rerank.assert_called_once()
        citations.assemble_citations.assert_called_once()

        assert isinstance(result, RetrievalResult)
        assert len(result.hits) == 1

    def test_filters_passed_through(self) -> None:
        """Explicit filters are forwarded to vector store."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

        filters = SearchFilters(folder_filter="/my/folder")
        engine.search("test", filters=filters)

        dense_call_filters = vs.query_dense.call_args[0][1]
        assert dense_call_filters.folder_filter == "/my/folder"

        keyword_call_filters = vs.query_keyword.call_args[0][1]
        assert keyword_call_filters.folder_filter == "/my/folder"

    def test_debug_mode_includes_timing(self) -> None:
        """Debug mode populates debug_info with timing data."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

        result = engine.search("test query", debug=True)

        assert result.debug_info is not None
        assert "embed_ms" in result.debug_info
        assert "dense_ms" in result.debug_info
        assert "keyword_ms" in result.debug_info
        assert "rerank_ms" in result.debug_info
        assert "total_ms" in result.debug_info
        assert "query_classification" in result.debug_info

    def test_debug_false_no_debug_info(self) -> None:
        """Without debug, debug_info is None."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

        result = engine.search("test query", debug=False)
        assert result.debug_info is None

    def test_custom_top_k(self) -> None:
        """Custom top_k is passed to reranker."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

        engine.search("test", top_k=5)

        reranker_call_top_k = reranker.rerank.call_args[0][2]
        assert reranker_call_top_k == 5

    def test_query_classification_in_result(self) -> None:
        """Result includes query classification."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

        result = engine.search("what is this")
        assert result.query_classification == "broad"

    def test_async_search_wraps_sync(self) -> None:
        """async_search dispatches to search via asyncio.to_thread."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

        result = asyncio.run(engine.async_search("test query"))

        assert isinstance(result, RetrievalResult)
        embedder.embed_query.assert_called_once()
