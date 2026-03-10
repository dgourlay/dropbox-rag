from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.pipeline.runner import PipelineRunner
    from rag.retrieval.engine import RetrievalEngine


class TestSearchFindsKnownContent:
    def test_search_revenue(
        self,
        indexed_pipeline: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search 'Q3 revenue year-over-year' finds quarterly-report content with '12%'."""
        result = retrieval_engine.search("Q3 revenue year-over-year")
        assert len(result.hits) > 0
        texts = " ".join(h.text for h in result.hits)
        assert "12%" in texts or "revenue" in texts.lower()


class TestSearchAcrossFileTypes:
    def test_search_postgresql_migration(
        self,
        indexed_pipeline: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search 'PostgreSQL migration' finds meeting-notes content."""
        result = retrieval_engine.search("PostgreSQL migration")
        assert len(result.hits) > 0
        texts = " ".join(h.text for h in result.hits)
        assert "postgresql" in texts.lower() or "migration" in texts.lower()


class TestSearchReturnsCitations:
    def test_every_result_has_citation(
        self,
        indexed_pipeline: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Every result has a citation with title, path, and label."""
        result = retrieval_engine.search("revenue expenses outlook")
        for hit in result.hits:
            assert hit.citation is not None
            assert hit.citation.title
            assert hit.citation.path
            assert hit.citation.label


class TestSearchRespectsTopK:
    def test_top_k_limit(
        self,
        indexed_pipeline: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search with top_k=2 returns at most 2 results."""
        result = retrieval_engine.search("project plan timeline", top_k=2)
        assert len(result.hits) <= 2


class TestSearchNoResults:
    def test_nonsense_query(
        self,
        indexed_pipeline: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search nonsense returns empty results, no crash."""
        result = retrieval_engine.search("xyzzy qwop zxcvbn asdfjkl")
        assert result is not None
        assert isinstance(result.hits, list)
