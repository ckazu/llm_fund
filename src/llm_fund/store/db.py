"""SQLite connection handling and numbered-SQL-file migrations.

Migrations live in ``store/migrations/*.sql`` and are applied in filename
order exactly once, tracked in a ``schema_migrations`` bookkeeping table.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_SCHEMA_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL UNIQUE,
    applied_at  TEXT NOT NULL
)
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name and FK checks on."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def applied_migrations(conn: sqlite3.Connection) -> set[str]:
    """Return the set of migration filenames already applied to `conn`."""
    conn.execute(_SCHEMA_MIGRATIONS_TABLE)
    rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()
    return {row["filename"] for row in rows}


def apply_migrations(
    conn: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR
) -> list[str]:
    """Apply pending numbered `.sql` migrations in filename order.

    Returns the filenames that were newly applied (empty if already up to date).
    Safe to call repeatedly: already-applied files are skipped.
    """
    conn.execute(_SCHEMA_MIGRATIONS_TABLE)
    already_applied = applied_migrations(conn)

    newly_applied: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name in already_applied:
            continue
        conn.executescript(path.read_text())
        conn.execute(
            "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
            (path.name, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        newly_applied.append(path.name)
    return newly_applied


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Open `db_path` and bring it up to date with all pending migrations."""
    conn = connect(db_path)
    apply_migrations(conn)
    return conn
