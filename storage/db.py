"""
storage/db.py
SQLite database layer.

Tables:
  messages     every message seen in a watched chat (source='watch') or checked ad hoc (source='adhoc'),
               with its origin (channel, forward source, t.me link) for receipts and channel comparison
  flagged      one row per flagged message (UNIQUE message_id — re-analysis updates, never duplicates)
  watch_chats  chats where real-time monitoring is enabled

Path: $BOT_DB_PATH or <repo>/data/bot.db
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "bot.db"

SOURCE_WATCH = "watch"
SOURCE_ADHOC = "adhoc"


def db_path() -> Path:
    return Path(os.getenv("BOT_DB_PATH") or _DEFAULT_DB)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_MESSAGE_COLUMNS: dict[str, str] = {
    # column: declaration — added with ALTER TABLE when missing (migration for hackathon-era DBs)
    "chat_title": "TEXT",
    "tg_message_id": "INTEGER",
    "sender_chat_id": "INTEGER",
    "sender_chat_title": "TEXT",
    "fwd_chat_id": "INTEGER",
    "fwd_chat_title": "TEXT",
    "fwd_chat_username": "TEXT",
    "fwd_sender_name": "TEXT",
    "link": "TEXT",
    "source": "TEXT NOT NULL DEFAULT 'watch'",
    "lang": "TEXT",
    "embedding": "BLOB",  # float32 query vector from the retrieval backend, for clustering
}


def init_db() -> None:
    conn = get_connection()
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     INTEGER NOT NULL,
                user_id     INTEGER,
                username    TEXT,
                text        TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS flagged (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id      INTEGER NOT NULL REFERENCES messages(id),
                chat_id         INTEGER NOT NULL,
                narrative_label TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                cluster_id      TEXT,
                backend         TEXT,
                flagged_at      TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS watch_chats (
                chat_id     INTEGER PRIMARY KEY,
                enabled_at  TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
            CREATE INDEX IF NOT EXISTS idx_flagged_chat  ON flagged(chat_id);
            CREATE INDEX IF NOT EXISTS idx_flagged_cluster ON flagged(cluster_id);
            """
        )
        _ensure_columns(conn, "messages", _MESSAGE_COLUMNS)
        _ensure_columns(conn, "flagged", {"backend": "TEXT"})
        # Collapse duplicates left by the hackathon version, then enforce uniqueness.
        conn.execute("DELETE FROM flagged WHERE id NOT IN (SELECT MAX(id) FROM flagged GROUP BY message_id)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_flagged_message ON flagged(message_id)")
    conn.close()
    logger.info("Database initialised at %s", db_path())


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


# ── Messages ─────────────────────────────────────────────────────────────────


def save_message(
    chat_id: int,
    user_id: int | None,
    username: str | None,
    text: str,
    *,
    source: str = SOURCE_WATCH,
    chat_title: str | None = None,
    tg_message_id: int | None = None,
    sender_chat_id: int | None = None,
    sender_chat_title: str | None = None,
    fwd_chat_id: int | None = None,
    fwd_chat_title: str | None = None,
    fwd_chat_username: str | None = None,
    fwd_sender_name: str | None = None,
    link: str | None = None,
    lang: str | None = None,
) -> int:
    """Insert a message and return its row id."""
    conn = get_connection()
    with conn:
        cur = conn.execute(
            """INSERT INTO messages
               (chat_id, chat_title, tg_message_id, user_id, username, sender_chat_id, sender_chat_title,
                fwd_chat_id, fwd_chat_title, fwd_chat_username, fwd_sender_name, link, source, lang,
                text, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                chat_id,
                chat_title,
                tg_message_id,
                user_id,
                username,
                sender_chat_id,
                sender_chat_title,
                fwd_chat_id,
                fwd_chat_title,
                fwd_chat_username,
                fwd_sender_name,
                link,
                source,
                lang,
                text,
                _now(),
            ),
        )
        row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def save_embedding(message_id: int, vector: list[float] | None) -> None:
    """Store the message's embedding (float32 little-endian bytes); no-op when the backend has none."""
    if not vector:
        return
    import array

    blob = array.array("f", vector).tobytes()
    conn = get_connection()
    with conn:
        conn.execute("UPDATE messages SET embedding=? WHERE id=?", (blob, message_id))
    conn.close()


def get_recent_messages(chat_id: int, limit: int = 10) -> list[sqlite3.Row]:
    """Most recent watched messages for a chat, oldest first. Ad-hoc checks are excluded."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? AND source=? ORDER BY id DESC LIMIT ?",
        (chat_id, SOURCE_WATCH, limit),
    ).fetchall()
    conn.close()
    return list(reversed(rows))


# ── Flagged ──────────────────────────────────────────────────────────────────


def save_flagged(
    message_id: int,
    chat_id: int,
    narrative_label: str,
    confidence: float,
    cluster_id: str | None,
    backend: str | None = None,
) -> None:
    """Record (or refresh) a propaganda hit. One row per message."""
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT INTO flagged (message_id, chat_id, narrative_label, confidence, cluster_id, backend, flagged_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(message_id) DO UPDATE SET
                 narrative_label=excluded.narrative_label,
                 confidence=excluded.confidence,
                 cluster_id=excluded.cluster_id,
                 backend=excluded.backend,
                 flagged_at=excluded.flagged_at""",
            (message_id, chat_id, narrative_label, confidence, cluster_id, backend, _now()),
        )
    conn.close()


def unflag(message_id: int) -> None:
    """Remove a flag (re-analysis said clean)."""
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM flagged WHERE message_id=?", (message_id,))
    conn.close()


_FLAGGED_SELECT = """
    SELECT f.*, m.text, m.username, m.timestamp, m.chat_title, m.sender_chat_title,
           m.fwd_chat_title, m.fwd_chat_username, m.fwd_sender_name, m.link, m.source
    FROM flagged f JOIN messages m ON f.message_id = m.id
"""


def get_flagged_for_chat(chat_id: int, limit: int = 20) -> list[sqlite3.Row]:
    """Most recent flagged messages for a chat, newest first."""
    conn = get_connection()
    rows = conn.execute(
        _FLAGGED_SELECT + " WHERE f.chat_id=? ORDER BY f.id DESC LIMIT ?", (chat_id, limit)
    ).fetchall()
    conn.close()
    return rows


def get_clusters_for_chat(chat_id: int) -> dict[str, list[sqlite3.Row]]:
    """Group flagged messages by cluster_id (narrative_label as fallback), newest first inside a group."""
    clusters: dict[str, list[sqlite3.Row]] = {}
    for row in get_flagged_for_chat(chat_id, limit=500):
        clusters.setdefault(row["cluster_id"] or row["narrative_label"], []).append(row)
    return clusters


# ── Watch mode ───────────────────────────────────────────────────────────────


def enable_watch(chat_id: int) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO watch_chats (chat_id, enabled_at) VALUES (?,?)", (chat_id, _now())
        )
    conn.close()


def disable_watch(chat_id: int) -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM watch_chats WHERE chat_id=?", (chat_id,))
    conn.close()


def is_watch_enabled(chat_id: int) -> bool:
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM watch_chats WHERE chat_id=?", (chat_id,)).fetchone()
    conn.close()
    return row is not None
