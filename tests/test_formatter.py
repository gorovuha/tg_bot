from bot.formatter import (
    format_analyze_result,
    format_clusters,
    format_flag_alert,
    format_report,
    source_line,
)
from services.classifier import ClassificationResult, Receipt


def test_label_and_text_are_escaped():
    r = ClassificationResult(True, 0.8, "<b>evil</b>", "<c>", "mock")
    out = format_flag_alert("<script>alert(1)</script>", "bob", r)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<b>evil</b>" not in out and "&lt;b&gt;evil" in out
    assert "mock classifier" in out


def test_no_mock_note_for_real_backend():
    r = ClassificationResult(False, 0.9, "None", None, "retrieval")
    assert "mock classifier" not in format_analyze_result("hi", r)


def test_evidence_rendered():
    r = ClassificationResult(
        True,
        0.8,
        "X",
        None,
        "retrieval",
        evidence=[Receipt("Case 1", "https://euvsdisinfo.eu/r/1", 0.87, "disproof")],
    )
    out = format_analyze_result("t", r)
    assert "Receipts" in out and 'href="https://euvsdisinfo.eu/r/1"' in out and "87%" in out


def test_source_line():
    row = {
        "chat_title": "Chan",
        "fwd_chat_title": "Src & Co",
        "fwd_chat_username": "src",
        "link": "https://t.me/c/1/2",
    }
    s = source_line(row)
    assert "Chan" in s and "Src &amp; Co" in s and 'href="https://t.me/src"' in s and "post" in s
    assert source_line({}) == ""


ROW = {
    "username": "u",
    "timestamp": "2026-04-21T10:00:00+00:00",
    "narrative_label": "A",
    "cluster_id": "c",
    "confidence": 0.8,
    "text": "hello",
    "backend": "mock",
    "chat_title": "Chan",
    "sender_chat_title": None,
    "fwd_chat_title": None,
    "fwd_chat_username": None,
    "fwd_sender_name": None,
    "link": None,
}


def test_report_channel_post_shows_channel_not_unknown():
    row = {**ROW, "username": None, "sender_chat_title": "My Channel"}
    out = format_report([row])
    assert "My Channel" in out and "unknown" not in out


def test_report_and_clusters():
    assert "21 Apr 10:00 UTC" in format_report([ROW])
    out = format_clusters({"c": [ROW, ROW]})
    assert "2 hits" in out and "Chan" in out


def test_formatters_accept_sqlite_rows(fresh_db):
    """Regression: sqlite3.Row has no .get(); /report and /analyze N crashed on real rows."""
    db = fresh_db
    mid = db.save_message(
        -1, None, "u", "nato text", chat_title="Chan", fwd_chat_title="Src", link="https://t.me/c/1/1"
    )
    db.save_flagged(mid, -1, "A", 0.8, "c", "mock")
    rows = db.get_flagged_for_chat(-1)
    assert "Chan" in source_line(rows[0]) and "Src" in source_line(rows[0])
    assert "Chan" in format_report(rows)
    assert "Chan" in format_clusters({"c": rows})
    r = ClassificationResult(True, 0.8, "A", "c", "mock")
    assert "Chan" in format_flag_alert("t", "u", r, db.get_recent_messages(-1)[0])
