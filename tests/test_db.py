def test_flag_is_upserted_not_duplicated(fresh_db):
    db = fresh_db
    mid = db.save_message(1, 2, "u", "text")
    db.save_flagged(mid, 1, "A", 0.8, "c1", "mock")
    db.save_flagged(mid, 1, "B", 0.9, "c2", "mock")
    rows = db.get_flagged_for_chat(1)
    assert len(rows) == 1
    assert rows[0]["narrative_label"] == "B" and rows[0]["confidence"] == 0.9


def test_unflag(fresh_db):
    db = fresh_db
    mid = db.save_message(1, None, None, "t")
    db.save_flagged(mid, 1, "A", 0.8, None)
    db.unflag(mid)
    assert db.get_flagged_for_chat(1) == []


def test_adhoc_excluded_from_recent(fresh_db):
    db = fresh_db
    db.save_message(1, None, None, "watched")
    db.save_message(1, None, None, "adhoc", source=db.SOURCE_ADHOC)
    rows = db.get_recent_messages(1)
    assert [r["text"] for r in rows] == ["watched"]


def test_origin_fields_roundtrip(fresh_db):
    db = fresh_db
    mid = db.save_message(
        -100123,
        None,
        None,
        "post",
        chat_title="Chan",
        tg_message_id=7,
        fwd_chat_title="Src",
        fwd_chat_username="src",
        link="https://t.me/chan/7",
    )
    db.save_flagged(mid, -100123, "A", 0.7, "c")
    row = db.get_flagged_for_chat(-100123)[0]
    assert row["chat_title"] == "Chan" and row["fwd_chat_username"] == "src" and row["link"].endswith("/7")


def test_migration_adds_columns_to_old_schema(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
            user_id INTEGER, username TEXT, text TEXT NOT NULL, timestamp TEXT NOT NULL);
        CREATE TABLE flagged (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER NOT NULL,
            chat_id INTEGER NOT NULL, narrative_label TEXT NOT NULL, confidence REAL NOT NULL,
            cluster_id TEXT, flagged_at TEXT NOT NULL);
        CREATE TABLE watch_chats (chat_id INTEGER PRIMARY KEY, enabled_at TEXT NOT NULL);
        INSERT INTO messages (chat_id, text, timestamp) VALUES (1, 'x', 't');
        INSERT INTO flagged (message_id, chat_id, narrative_label, confidence, flagged_at) VALUES (1,1,'A',0.5,'t');
        INSERT INTO flagged (message_id, chat_id, narrative_label, confidence, flagged_at) VALUES (1,1,'B',0.6,'t');
    """)
    conn.commit()
    conn.close()
    monkeypatch.setenv("BOT_DB_PATH", str(path))
    from storage import db

    db.init_db()
    rows = db.get_flagged_for_chat(1)
    assert len(rows) == 1 and rows[0]["narrative_label"] == "B"
    db.save_message(1, None, None, "new", link="l")  # new columns usable
