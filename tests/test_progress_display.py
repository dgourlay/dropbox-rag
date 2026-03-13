from __future__ import annotations

from click.testing import CliRunner

from rag.cli import _ProgressDisplay
from rag.types import ProcessingOutcome


class TestProgressDisplay:
    """Tests for the line-by-line progress display."""

    def test_on_start_shows_parsing(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            d = _ProgressDisplay(total=5)
            d.on_start(1, 5, "test.pdf")
        # No exception means it printed fine

    def test_on_done_shows_result(self) -> None:
        d = _ProgressDisplay(total=5)
        d.on_done(1, 5, "test.pdf", ProcessingOutcome.INDEXED, "4 chunks")

    def test_consistent_file_index(self) -> None:
        """Same file index format in both start and done."""
        d = _ProgressDisplay(total=20)
        # Just ensure no crash with consistent index
        d.on_start(7, 20, "report.pdf")
        d.on_done(7, 20, "report.pdf", ProcessingOutcome.INDEXED, "10 chunks")

    def test_long_name_truncated(self) -> None:
        long_name = "a" * 60 + ".pdf"
        d = _ProgressDisplay(total=5)
        fitted = d._fit_name(long_name)
        assert "…" in fitted
        assert len(fitted) == d._NAME_W

    def test_short_name_padded(self) -> None:
        d = _ProgressDisplay(total=5)
        fitted = d._fit_name("short.pdf")
        assert "short.pdf" in fitted
        assert len(fitted) == d._NAME_W

    def test_all_outcomes_render(self) -> None:
        """Every ProcessingOutcome produces output without error."""
        outcomes = [
            (ProcessingOutcome.INDEXED, "4 chunks"),
            (ProcessingOutcome.UNCHANGED, "content unchanged"),
            (ProcessingOutcome.DUPLICATE, "duplicate of abc"),
            (ProcessingOutcome.DELETED, "removed from index"),
            (ProcessingOutcome.ERROR, "parse failed"),
        ]
        d = _ProgressDisplay(total=5)
        for outcome, detail in outcomes:
            d.on_done(1, 5, "test.pdf", outcome, detail)

    def test_finalize_is_noop(self) -> None:
        d = _ProgressDisplay(total=5)
        d.finalize()  # Should not raise
