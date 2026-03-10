from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "migrations"


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> None:
    """Execute all .sql migration files idempotently.

    Files are sorted by name so 001_*.sql runs before 002_*.sql.
    Uses CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS
    in the SQL files themselves for idempotency.
    """
    directory = migrations_dir or _MIGRATIONS_DIR
    sql_files = sorted(directory.glob("*.sql"))

    for sql_file in sql_files:
        sql = sql_file.read_text(encoding="utf-8")
        conn.executescript(sql)
