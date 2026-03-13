from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "migrations"


def _get_applied_migrations(conn: sqlite3.Connection) -> set[str]:
    """Get the set of already-applied migration filenames."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations ("
        "  filename TEXT PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    rows = conn.execute("SELECT filename FROM _migrations").fetchall()
    return {row[0] for row in rows}


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> None:
    """Execute all .sql migration files idempotently.

    Files are sorted by name so 001_*.sql runs before 002_*.sql.
    Tracks applied migrations in a _migrations table to avoid re-running
    non-idempotent DDL (like ALTER TABLE RENAME COLUMN).
    """
    directory = migrations_dir or _MIGRATIONS_DIR
    sql_files = sorted(directory.glob("*.sql"))

    applied = _get_applied_migrations(conn)

    for sql_file in sql_files:
        if sql_file.name in applied:
            continue
        sql = sql_file.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO _migrations (filename) VALUES (?)",
            (sql_file.name,),
        )
        conn.commit()
