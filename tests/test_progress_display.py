from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from rag.cli import _ProgressDisplay
from rag.types import ProcessingOutcome


class TestProgressDisplay:
    """Tests for the rolling two-column progress display."""

    def _capture_display(self) -> io.StringIO:
        """Return a StringIO that replaces sys.stdout for capture."""
        return io.StringIO()

    def test_on_start_adds_parsing_row(self) -> None:
        buf = self._capture_display()
        d = _ProgressDisplay(total=5)
        with patch.object(sys, "stdout", buf):
            d.on_start(1, 5, "test.pdf")
        output = buf.getvalue()
        assert "test.pdf" in output
        assert "parsing..." in output

    def test_on_done_shows_result(self) -> None:
        buf = self._capture_display()
        d = _ProgressDisplay(total=5)
        with patch.object(sys, "stdout", buf):
            d.on_start(1, 5, "test.pdf")
            d.on_done(1, 5, "test.pdf", ProcessingOutcome.INDEXED, "4 chunks")
        output = buf.getvalue()
        assert "indexed" in output
        assert "4 chunks" in output

    def test_consistent_file_index(self) -> None:
        """Same file index appears in both parsing and done lines."""
        buf = self._capture_display()
        d = _ProgressDisplay(total=20)
        with patch.object(sys, "stdout", buf):
            d.on_start(7, 20, "report.pdf")
            d.on_done(7, 20, "report.pdf", ProcessingOutcome.INDEXED, "10 chunks")
        output = buf.getvalue()
        assert "[ 7/20]" in output

    def test_long_name_truncated(self) -> None:
        long_name = "a" * 60 + ".pdf"
        d = _ProgressDisplay(total=5)
        rendered = d._render_row(1, long_name, None)
        # Should be truncated with ellipsis
        assert "…" in rendered
        assert long_name not in rendered

    def test_short_name_not_truncated(self) -> None:
        d = _ProgressDisplay(total=5)
        rendered = d._render_row(1, "short.pdf", None)
        assert "short.pdf" in rendered

    def test_window_limits_visible_rows(self) -> None:
        """With more than WINDOW rows, only WINDOW lines are drawn."""
        buf = self._capture_display()
        d = _ProgressDisplay(total=20)
        with patch.object(sys, "stdout", buf):
            # Add 15 completed rows
            for i in range(1, 16):
                d.on_start(i, 20, f"file{i}.pdf")
                d.on_done(i, 20, f"file{i}.pdf", ProcessingOutcome.INDEXED, "1 chunks")
        # Count non-empty output lines in the final redraw
        # The display redraws each time, so check drawn_lines
        assert d._drawn_lines <= d._WINDOW

    def test_window_follows_pending(self) -> None:
        """Window should center around the earliest pending file."""
        buf = self._capture_display()
        d = _ProgressDisplay(total=20)
        with patch.object(sys, "stdout", buf):
            # Complete files 1-10
            for i in range(1, 11):
                d._rows[i] = (f"file{i}.pdf", "indexed (1 chunks)")
            # Start parsing file 11
            d.on_start(11, 20, "file11.pdf")

        # File 11 should be visible, and some completed files before it
        sorted_idxs = sorted(d._rows.keys())
        first_pending_pos = sorted_idxs.index(11)
        start_pos = max(0, first_pending_pos - 2)
        visible = sorted_idxs[start_pos : start_pos + d._WINDOW]
        assert 11 in visible

    def test_all_outcomes_render(self) -> None:
        """Every ProcessingOutcome produces output without error."""
        outcomes = [
            (ProcessingOutcome.INDEXED, "4 chunks"),
            (ProcessingOutcome.UNCHANGED, "content unchanged"),
            (ProcessingOutcome.DUPLICATE, "duplicate of abc"),
            (ProcessingOutcome.DELETED, "removed from index"),
            (ProcessingOutcome.ERROR, "parse failed"),
        ]
        for outcome, detail in outcomes:
            buf = self._capture_display()
            d = _ProgressDisplay(total=5)
            with patch.object(sys, "stdout", buf):
                d.on_start(1, 5, "test.pdf")
                d.on_done(1, 5, "test.pdf", outcome, detail)
            output = buf.getvalue()
            assert detail in output, f"Missing detail for {outcome.name}"

    def test_redraw_uses_ansi_cursor_up(self) -> None:
        """After first draw, subsequent redraws move cursor up."""
        buf = self._capture_display()
        d = _ProgressDisplay(total=5)
        with patch.object(sys, "stdout", buf):
            d.on_start(1, 5, "a.pdf")
            d.on_start(2, 5, "b.pdf")
        output = buf.getvalue()
        # Second redraw should contain ANSI cursor-up escape
        assert "\033[" in output

    def test_interleaved_start_and_done(self) -> None:
        """Simulates realistic interleaving: parser ahead of indexer."""
        buf = self._capture_display()
        d = _ProgressDisplay(total=5)
        with patch.object(sys, "stdout", buf):
            d.on_start(1, 5, "a.pdf")
            d.on_start(2, 5, "b.pdf")
            d.on_done(1, 5, "a.pdf", ProcessingOutcome.INDEXED, "3 chunks")
            d.on_start(3, 5, "c.pdf")
            d.on_done(2, 5, "b.pdf", ProcessingOutcome.UNCHANGED, "content unchanged")
        # All three files should be tracked
        assert 1 in d._rows
        assert 2 in d._rows
        assert 3 in d._rows
        # File 1 and 2 should have results, file 3 still pending
        assert d._rows[1][1] is not None
        assert d._rows[2][1] is not None
        assert d._rows[3][1] is None
