import array
from datetime import UTC, datetime, timedelta

import numpy as np

from bot.formatter import format_channels, format_cluster_map
from services.clustering import channel_report, cluster_messages, sparkline


def _blob(v):
    v = np.asarray(v, dtype=np.float32)
    return array.array("f", (v / np.linalg.norm(v)).tolist()).tobytes()


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def row(text, vec, label, nid, chan, hours_ago=0, fwd=None, fwd_user=None, conf=0.9):
    return {
        "text": text,
        "embedding": _blob(vec),
        "narrative_label": label,
        "cluster_id": nid,
        "chat_title": chan,
        "sender_chat_title": None,
        "fwd_chat_title": fwd,
        "fwd_chat_username": fwd_user,
        "fwd_sender_name": None,
        "confidence": conf,
        "timestamp": (NOW - timedelta(hours=hours_ago)).isoformat(timespec="seconds"),
    }


ROWS = [
    row("biolabs 1", [1, 0, 0], "Biolabs", "biolabs", "Chan A", 1),
    row("biolabs 2", [0.98, 0.05, 0], "Biolabs", "biolabs", "Chan B", 3),
    row("biolabs 3", [0.97, 0, 0.05], "Biolabs", "biolabs", "Chan A", 30),
    row("crimea", [0, 1, 0], "Crimea", "crimea_legal", "Chan B", 2),
    row("nato fwd", [0, 0, 1], "NATO", "nato_threat", "Chan A", 5, fwd="Kremlin TV", fwd_user="KremlinTV"),
]


def test_clusters_by_embedding_largest_first():
    cl = cluster_messages(ROWS)
    assert [c.size for c in cl] == [3, 1, 1]
    big = cl[0]
    assert big.label == "Biolabs" and big.narrative_id == "biolabs" and big.id == 1
    assert big.sources == {"Chan A": 2, "Chan B": 1}
    assert big.rows[0]["text"] == "biolabs 1"  # newest first
    assert big.first_seen < big.last_seen


def test_fallback_grouping_without_embeddings():
    rows = [dict(r, embedding=None) for r in ROWS]
    cl = cluster_messages(rows)
    assert {c.narrative_id: c.size for c in cl} == {"biolabs": 3, "crimea_legal": 1, "nato_threat": 1}


def test_daily_counts_and_sparkline():
    cl = cluster_messages(ROWS)[0]
    counts = cl.daily_counts(days=3, today=NOW.date())
    assert sum(counts) == 3 and counts[-1] == 2  # two today, one yesterday
    assert sparkline([0, 1, 2]) == "▁▅█" and sparkline([0, 0]) == "▁▁"


def test_channel_report_pairs_sync_and_known():
    rep = channel_report(ROWS, known_channels={"kremlintv": 7})
    names = [p.name for p in rep.channels]
    assert names[:2] == ["Chan A", "Chan B"] and "Kremlin TV" in names  # forward origin is its own source
    kremlin = next(p for p in rep.channels if p.name == "Kremlin TV")
    assert kremlin.known_case_hits == 7 and kremlin.narratives == {"nato_threat": 1}
    a, b, n, labels = rep.pairs[0]
    assert {a, b} == {"Chan A", "Chan B"} and n == 1 and labels == ["Biolabs"]
    assert rep.synchronous and rep.synchronous[0][0] == "Biolabs"
    assert rep.synchronous[0][1] == ["Chan A", "Chan B"] and rep.synchronous[0][2] == timedelta(hours=2)


def test_empty_inputs():
    assert cluster_messages([]) == []
    rep = channel_report([])
    assert rep.channels == [] and "No channel data" in format_channels(rep)
    assert "No clusters" in format_cluster_map([])


def test_formatters_render():
    cl = cluster_messages(ROWS)
    out = format_cluster_map(cl, days=7)
    assert "Biolabs" in out and "Chan A (2)" in out and "<code>" in out
    rep = channel_report(ROWS, known_channels={"kremlintv": 7})
    out = format_channels(rep, {"biolabs": "US biolabs"})
    assert "US biolabs ×2" in out and "spreader in 7" in out and "Synchronous" in out and "↔" in out
