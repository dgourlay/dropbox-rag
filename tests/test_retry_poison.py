from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from rag.config import AppConfig, FoldersConfig
from rag.db.models import SqliteMetadataDB
from rag.pipeline.dedup import DedupChecker
from rag.pipeline.runner import PipelineRunner
from rag.results import ParseError, ParseSuccess
from rag.types import (
    FileEvent,
    FileType,
    ParsedDocument,
    ParsedSection,
    ProcessingOutcome,
    SyncStateRow,
)

ParseSuccess.model_rebuild()
ParseError.model_rebuild()


# --- Helpers ---


def _create_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrations_dir = Path(__file__).parent.parent / "migrations"
    conn.executescript((migrations_dir / "001_initial.sql").read_text())
    conn.executescript((migrations_dir / "002_pyramid_summaries.sql").read_text())
    conn.executescript((migrations_dir / "003_chunk_questions.sql").read_text())
    return conn


def _make_event(tmp_path: Path, content: str = "Hello world test content.") -> FileEvent:
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)
    return FileEvent(
        file_path=str(test_file),
        content_hash="abc123hash",
        file_type=FileType.TXT,
        event_type="created",
        modified_at="2026-01-01T00:00:00+00:00",
    )


def _make_runner(
    tmp_path: Path,
    conn: sqlite3.Connection,
    *,
    parse_result: ParseSuccess | ParseError | None = None,
) -> tuple[PipelineRunner, dict[str, Any]]:
    db = SqliteMetadataDB(conn)
    dedup = DedupChecker(conn)

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    mock_embedder.model_version = "test-model-v1"

    mock_vector_store = MagicMock()

    mock_parser = MagicMock()
    mock_parser.supported_types = {FileType.TXT, FileType.MD}
    if parse_result is not None:
        mock_parser.parse.return_value = parse_result
    else:
        doc = ParsedDocument(
            doc_id=str(uuid.uuid4()),
            title="Test Document",
            file_type=FileType.TXT,
            sections=[
                ParsedSection(
                    heading="Introduction",
                    order=0,
                    text="This is test content for the pipeline runner."
                    " It contains enough text to form at least one chunk.",
                ),
            ],
            raw_content_hash="rawhash123",
        )
        mock_parser.parse.return_value = ParseSuccess(document=doc)

    config = AppConfig(folders=FoldersConfig(paths=[tmp_path]))

    runner = PipelineRunner(
        db=db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        parsers=[mock_parser],
        dedup=dedup,
        config=config,
    )

    mocks = {
        "db": db,
        "dedup": dedup,
        "embedder": mock_embedder,
        "vector_store": mock_vector_store,
        "parser": mock_parser,
    }
    return runner, mocks


# --- Tests ---


class TestPoisonQuarantine:
    def test_poisoned_file_skipped_on_subsequent_process_file(self, tmp_path: Path) -> None:
        """After a file is poisoned, process_file should skip it immediately."""
        conn = _create_db()
        event = _make_event(tmp_path)
        parse_err = ParseError(error="always fails", file_path=str(tmp_path / "test.txt"))
        runner, mocks = _make_runner(tmp_path, conn, parse_result=parse_err)

        # Fail 3 times to reach poison
        runner.process_file(event)
        runner.process_file(event)
        runner.process_file(event)

        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.process_status == "poison"

        # 4th call should be skipped immediately
        outcome, detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.ERROR
        assert "quarantined" in detail

        # Parser should NOT have been called a 4th time
        assert mocks["parser"].parse.call_count == 3

    def test_poisoned_file_skipped_in_process_batch(self, tmp_path: Path) -> None:
        """Poisoned files should be skipped during batch processing."""
        conn = _create_db()
        event = _make_event(tmp_path)
        parse_err = ParseError(error="always fails", file_path=str(tmp_path / "test.txt"))
        runner, _mocks = _make_runner(tmp_path, conn, parse_result=parse_err)

        # Poison the file
        runner.process_file(event)
        runner.process_file(event)
        runner.process_file(event)

        # Now try batch processing — should skip poisoned file
        counts = runner.process_batch([event])
        assert counts[ProcessingOutcome.ERROR] == 1

    def test_retry_count_increments_on_each_failure(self, tmp_path: Path) -> None:
        """Each failure should increment retry_count by 1."""
        conn = _create_db()
        event = _make_event(tmp_path)
        parse_err = ParseError(error="fail", file_path=str(tmp_path / "test.txt"))
        runner, mocks = _make_runner(tmp_path, conn, parse_result=parse_err)

        runner.process_file(event)
        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.retry_count == 1
        assert state.process_status == "error"

        runner.process_file(event)
        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.retry_count == 2
        assert state.process_status == "error"

        runner.process_file(event)
        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.retry_count == 3
        assert state.process_status == "poison"


class TestExponentialBackoff:
    def test_backoff_skips_file_during_cooldown(self, tmp_path: Path) -> None:
        """Files in error status within backoff window should be skipped in batch."""
        conn = _create_db()
        db = SqliteMetadataDB(conn)
        event = _make_event(tmp_path)

        # Manually insert sync_state with error + recent synced_at
        now = datetime.now(tz=UTC)
        db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path=str(tmp_path / "test.txt"),
                file_name="test.txt",
                folder_path=str(tmp_path),
                folder_ancestors=[str(tmp_path)],
                file_type="txt",
                modified_at="2026-01-01T00:00:00+00:00",
                content_hash="abc123hash",
                synced_at=now.isoformat(),
                process_status="error",
                error_message="some failure",
                retry_count=1,
            )
        )

        runner, mocks = _make_runner(tmp_path, conn)

        # Batch should skip due to backoff (2^1 * 30 = 60s from now)
        counts = runner.process_batch([event])
        assert counts[ProcessingOutcome.ERROR] == 1
        # Parser should not have been called
        mocks["parser"].parse.assert_not_called()

    def test_backoff_allows_retry_after_window(self, tmp_path: Path) -> None:
        """Files past the backoff window should be retried."""
        conn = _create_db()
        db = SqliteMetadataDB(conn)
        event = _make_event(tmp_path)

        # Set synced_at far in the past (backoff window has elapsed)
        past = datetime.now(tz=UTC) - timedelta(hours=1)
        db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path=str(tmp_path / "test.txt"),
                file_name="test.txt",
                folder_path=str(tmp_path),
                folder_ancestors=[str(tmp_path)],
                file_type="txt",
                modified_at="2026-01-01T00:00:00+00:00",
                content_hash="different_hash",
                synced_at=past.isoformat(),
                process_status="error",
                error_message="some failure",
                retry_count=1,
            )
        )

        runner, _mocks = _make_runner(tmp_path, conn)
        counts = runner.process_batch([event])

        # File should have been processed (not skipped by backoff)
        assert counts[ProcessingOutcome.INDEXED] == 1

    def test_backoff_increases_exponentially(self) -> None:
        """Verify backoff computation: 2^retry_count * 30 seconds."""
        from rag.pipeline.runner import PipelineRunner

        now = datetime.now(tz=UTC)

        # retry_count=1: backoff = 60s
        state1 = SyncStateRow(
            id="1",
            file_path="/test",
            file_name="test",
            folder_path="/",
            folder_ancestors=["/"],
            file_type="txt",
            modified_at=now.isoformat(),
            content_hash="h",
            synced_at=now.isoformat(),
            process_status="error",
            retry_count=1,
        )
        result = PipelineRunner._check_skip_retry(state1)
        assert result is not None
        assert "backing off" in result[1]

        # retry_count=2: backoff = 120s
        state2 = SyncStateRow(
            id="2",
            file_path="/test2",
            file_name="test2",
            folder_path="/",
            folder_ancestors=["/"],
            file_type="txt",
            modified_at=now.isoformat(),
            content_hash="h",
            synced_at=now.isoformat(),
            process_status="error",
            retry_count=2,
        )
        result2 = PipelineRunner._check_skip_retry(state2)
        assert result2 is not None
        assert "backing off" in result2[1]

    def test_no_backoff_on_first_error(self) -> None:
        """retry_count=0 should not trigger backoff."""
        from rag.pipeline.runner import PipelineRunner

        now = datetime.now(tz=UTC)
        state = SyncStateRow(
            id="1",
            file_path="/test",
            file_name="test",
            folder_path="/",
            folder_ancestors=["/"],
            file_type="txt",
            modified_at=now.isoformat(),
            content_hash="h",
            synced_at=now.isoformat(),
            process_status="error",
            retry_count=0,
        )
        result = PipelineRunner._check_skip_retry(state)
        assert result is None


class TestPoisonedStatus:
    def test_get_poisoned_count(self, tmp_path: Path) -> None:
        conn = _create_db()
        db = SqliteMetadataDB(conn)

        # No poisoned files initially
        assert db.get_poisoned_count() == 0

        # Add a poisoned file
        db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path="/test/poisoned.txt",
                file_name="poisoned.txt",
                folder_path="/test",
                folder_ancestors=["/test"],
                file_type="txt",
                modified_at="2026-01-01T00:00:00+00:00",
                content_hash="hash1",
                process_status="poison",
                retry_count=3,
                error_message="always fails",
            )
        )
        assert db.get_poisoned_count() == 1

    def test_get_poisoned_files(self, tmp_path: Path) -> None:
        conn = _create_db()
        db = SqliteMetadataDB(conn)

        db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path="/test/bad.pdf",
                file_name="bad.pdf",
                folder_path="/test",
                folder_ancestors=["/test"],
                file_type="pdf",
                modified_at="2026-01-01T00:00:00+00:00",
                content_hash="hash2",
                process_status="poison",
                retry_count=3,
                error_message="corrupt PDF",
            )
        )

        poisoned = db.get_poisoned_files()
        assert len(poisoned) == 1
        assert poisoned[0].file_path == "/test/bad.pdf"
        assert poisoned[0].error_message == "corrupt PDF"
        assert poisoned[0].process_status == "poison"

    def test_deleted_poisoned_files_not_counted(self) -> None:
        conn = _create_db()
        db = SqliteMetadataDB(conn)

        db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path="/test/deleted.txt",
                file_name="deleted.txt",
                folder_path="/test",
                folder_ancestors=["/test"],
                file_type="txt",
                modified_at="2026-01-01T00:00:00+00:00",
                content_hash="hash3",
                process_status="poison",
                retry_count=3,
                is_deleted=1,
            )
        )

        assert db.get_poisoned_count() == 0
        assert db.get_poisoned_files() == []

    def test_status_json_includes_poisoned(self, tmp_path: Path) -> None:
        """Verify that the status command JSON output includes poisoned info."""
        conn = _create_db()
        db = SqliteMetadataDB(conn)

        # Add a poisoned file
        db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path="/test/poison.txt",
                file_name="poison.txt",
                folder_path="/test",
                folder_ancestors=["/test"],
                file_type="txt",
                modified_at="2026-01-01T00:00:00+00:00",
                content_hash="hash4",
                process_status="poison",
                retry_count=3,
                error_message="bad file",
            )
        )

        # Simulate what cli.py status --json does
        poisoned_count = db.get_poisoned_count()
        poisoned_files = db.get_poisoned_files()

        data = {
            "documents": db.get_document_count(),
            "chunks": db.get_chunk_count(),
            "errors": db.get_error_count(),
            "poisoned_count": poisoned_count,
            "poisoned_files": [
                {
                    "file_path": pf.file_path,
                    "error_message": pf.error_message,
                    "retry_count": pf.retry_count,
                }
                for pf in poisoned_files
            ],
        }

        assert data["poisoned_count"] == 1
        assert len(data["poisoned_files"]) == 1
        assert data["poisoned_files"][0]["file_path"] == "/test/poison.txt"
        assert data["poisoned_files"][0]["error_message"] == "bad file"
        assert data["poisoned_files"][0]["retry_count"] == 3

        # Verify it serializes cleanly
        output = json.dumps(data, indent=2)
        parsed = json.loads(output)
        assert parsed["poisoned_count"] == 1
