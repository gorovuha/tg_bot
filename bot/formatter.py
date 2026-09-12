"""
bot/formatter.py
Telegram HTML rendering. Every user- or backend-provided string goes through `esc()`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime
from html import escape

from services.classifier import MOCK_BACKEND, ClassificationResult

RULE = "━━━━━━━━━━━━━━━━━━━━━━━━━"
MOCK_NOTE = "⚠️ <i>mock classifier (keyword heuristics) — not a real model</i>"


def esc(value: object) -> str:
    return escape(str(value), quote=False)


def trim(text: str, max_len: int) -> str:
    """Trim and HTML-escape a text snippet."""
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return esc(text)


def _ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %H:%M UTC")
    except (TypeError, ValueError):
        return esc(iso)


def _conf_bar(confidence: float, length: int = 10) -> str:
    filled = max(0, min(length, round(confidence * length)))
    return "█" * filled + "░" * (length - filled)


def source_line(row: Mapping | sqlite3.Row) -> str:
    """'📡 Channel Title · fwd from X · link' built from message origin fields (missing keys tolerated)."""
    if isinstance(row, sqlite3.Row):
        row = dict(row)
    parts: list[str] = []
    origin = row.get("sender_chat_title") or row.get("chat_title")
    if origin:
        parts.append(f"<b>{esc(origin)}</b>")
    fwd = row.get("fwd_chat_title") or row.get("fwd_sender_name")
    if fwd:
        fwd_s = esc(fwd)
        if row.get("fwd_chat_username"):
            fwd_s = f'<a href="https://t.me/{esc(row["fwd_chat_username"])}">{fwd_s}</a>'
        parts.append(f"fwd from {fwd_s}")
    if row.get("link"):
        parts.append(f'<a href="{esc(row["link"])}">post</a>')
    return ("📡 " + "  ·  ".join(parts)) if parts else ""


def _backend_note(backend: str | None) -> str:
    return f"\n{MOCK_NOTE}" if backend == MOCK_BACKEND else ""


def format_evidence(result: ClassificationResult, max_items: int = 3) -> str:
    if not result.evidence:
        return ""
    lines = [
        "🧾 <b>Receipts:</b>" if result.is_propaganda else "🔎 <b>Nearest known case</b> (below threshold):"
    ]
    for r in result.evidence[:max_items]:
        title = f'<a href="{esc(r.url)}">{esc(r.title)}</a>' if r.url else esc(r.title)
        lines.append(f"  • {title} ({r.similarity:.0%})")
        if r.excerpt:
            lines.append(f"    <i>{trim(r.excerpt, 160)}</i>")
    return "\n".join(lines)


# ── Alerts / results ─────────────────────────────────────────────────────────


def format_flag_alert(
    text: str, username: str | None, result: ClassificationResult, source: Mapping | None = None
) -> str:
    """Inline alert posted right after a propaganda message is detected in watch mode."""
    user_str = f"@{esc(username)}" if username else "unknown user"
    cluster = f"🗂 Cluster: <code>{esc(result.cluster_id)}</code>\n" if result.cluster_id else ""
    src = source_line(source) if source else ""
    evidence = format_evidence(result)
    return (
        f"🚨 <b>PROPAGANDA DETECTED</b>\n{RULE}\n"
        f"👤 From: {user_str}\n"
        + (f"{src}\n" if src else "")
        + f"🏷 Narrative: <b>{esc(result.narrative_label)}</b>\n"
        f"{cluster}"
        f"📊 Confidence: {_conf_bar(result.confidence)} {result.confidence:.0%}\n"
        + (f"{evidence}\n" if evidence else "")
        + f'{RULE}\n💬 <i>"{trim(text, 200)}"</i>'
        f"{_backend_note(result.backend)}"
    )


def format_analyze_result(text: str, result: ClassificationResult) -> str:
    """Result card for an on-demand /analyze <text> check."""
    verdict = "🚨 <b>PROPAGANDA</b>" if result.is_propaganda else "✅ <b>CLEAN</b>"
    cluster = f"\n🗂 Cluster: <code>{esc(result.cluster_id)}</code>" if result.cluster_id else ""
    evidence = format_evidence(result)
    return (
        f"{verdict}\n{RULE}\n"
        f"🏷 Narrative: <b>{esc(result.narrative_label)}</b>{cluster}\n"
        f"📊 Confidence: {_conf_bar(result.confidence)} {result.confidence:.0%}\n"
        + (f"{evidence}\n" if evidence else "")
        + f'{RULE}\n📝 Analysed text:\n<i>"{trim(text, 300)}"</i>'
        f"{_backend_note(result.backend)}"
    )


def format_report(rows: list[Mapping]) -> str:
    if not rows:
        return "✅ <b>No propaganda found</b>\nNo flagged messages in this chat yet.\nUse /watch to start real-time monitoring."

    n = len(rows)
    lines = [f"📋 <b>PROPAGANDA REPORT</b>  ({n} hit{'s' if n != 1 else ''})\n{RULE}\n"]
    for i, row in enumerate(rows, 1):
        user_str = (
            f"@{esc(row['username'])}" if row["username"] else esc(row["sender_chat_title"] or "unknown")
        )
        cluster = f"🗂 <code>{esc(row['cluster_id'])}</code>  " if row["cluster_id"] else ""
        src = source_line(row)
        lines.append(
            f"<b>#{i}</b>  [{_ts(row['timestamp'])}]  {user_str}\n"
            + (f"{src}\n" if src else "")
            + f"🏷 <b>{esc(row['narrative_label'])}</b>  {cluster}({row['confidence']:.0%})\n"
            f'💬 <i>"{trim(row["text"], 120)}"</i>\n'
        )
    lines.append(f"{RULE}\nUse /cluster to see narrative groups.")
    if any(row["backend"] == MOCK_BACKEND for row in rows):
        lines.append(MOCK_NOTE)
    return "\n".join(lines)


def format_clusters(clusters: dict[str, list[Mapping]]) -> str:
    if not clusters:
        return "🗂 <b>No clusters yet.</b>\nEnable /watch and collect some messages first."

    total = sum(len(v) for v in clusters.values())
    n = len(clusters)
    lines = [
        f"🗺 <b>NARRATIVE CLUSTER MAP</b>  ({n} cluster{'s' if n != 1 else ''}  ·  {total} hits)\n{RULE}\n"
    ]
    for key, rows in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        avg_conf = sum(r["confidence"] for r in rows) / len(rows)
        label_counts: dict[str, int] = {}
        for r in rows:
            label_counts[r["narrative_label"]] = label_counts.get(r["narrative_label"], 0) + 1
        top_label = max(label_counts, key=label_counts.__getitem__)
        sources = {
            r["sender_chat_title"] or r["chat_title"]
            for r in rows
            if r["sender_chat_title"] or r["chat_title"]
        }
        src = f"   📡 {', '.join(esc(s) for s in sorted(sources))}\n" if sources else ""
        lines.append(
            f"🏷 <b>{esc(top_label)}</b>\n"
            f"   🗂 <code>{esc(key)}</code>  ·  {len(rows)} hit{'s' if len(rows) != 1 else ''}"
            f"  ·  avg {_conf_bar(avg_conf, 8)} {avg_conf:.0%}\n"
            f"{src}"
            f'   Latest: <i>"{trim(rows[0]["text"], 80)}"</i>\n'
        )
    lines.append(RULE)
    return "\n".join(lines)


def format_watch_on() -> str:
    return (
        "👁 <b>Watch mode ENABLED</b>\nI'll now monitor every message in this chat.\n"
        "Propaganda hits will be flagged immediately.\n\nUse /watch again to disable.  Use /report for a summary."
    )


def format_watch_off() -> str:
    return (
        "🔇 <b>Watch mode DISABLED</b>\nI've stopped monitoring this chat.\n"
        "Use /watch to re-enable.  Stored receipts still available via /report."
    )


# ── /cluster and /channels (phase 4) ─────────────────────────────────────────


def _span(td) -> str:
    h = td.total_seconds() / 3600
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.0f} h" if h < 48 else f"{h / 24:.0f} d"


def format_cluster_map(clusters, days: int = 14, scope: str = "this chat") -> str:
    """Embedding clusters of flagged messages with a per-day sparkline and source channels."""
    from services.clustering import sparkline

    if not clusters:
        return "🗂 <b>No clusters yet.</b>\nEnable /watch and collect some messages first."
    total = sum(c.size for c in clusters)
    n = len(clusters)
    lines = [
        (
            f"🗺 <b>NARRATIVE CLUSTER MAP</b> — {esc(scope)}\n"
            f"{n} cluster{'s' if n != 1 else ''}  ·  {total} flagged  ·  last {days} days per bar\n{RULE}\n"
        )
    ]
    for c in clusters[:12]:
        src = ", ".join(f"{esc(name)} ({cnt})" for name, cnt in c.sources.most_common(4))
        when = (
            f"{c.first_seen:%d %b}"
            if c.first_seen.date() == c.last_seen.date()
            else f"{c.first_seen:%d %b} → {c.last_seen:%d %b}"
        )
        lines.append(
            f"<b>#{c.id}  {esc(c.label)}</b>\n"
            f"   {c.size} msg{'s' if c.size != 1 else ''}  ·  avg {c.avg_confidence:.0%}  ·  {when}\n"
            f"   <code>{sparkline(c.daily_counts(days))}</code>\n"
            f"   📡 {src}\n"
            f'   <i>"{trim(c.rows[0]["text"], 90)}"</i>\n'
        )
    if n > 12:
        lines.append(f"… and {n - 12} smaller clusters\n")
    lines.append(RULE + "\nUse /channels to compare sources.")
    return "\n".join(lines)


def format_channels(report, label_of=None) -> str:
    """Channel × narrative profile, channel pairs sharing narratives, synchronous pushes."""
    label_of = label_of or {}
    if not report.channels:
        return "📡 <b>No channel data yet.</b>\nWatch a few channels (or forward posts from them) first."
    lines = [f"📡 <b>CHANNEL COMPARISON</b>  ({len(report.channels)} sources)\n{RULE}\n"]
    for p in report.channels[:10]:
        top = "  ·  ".join(f"{esc(label_of.get(k, k))} ×{v}" for k, v in p.narratives.most_common(3))
        known = (
            f"\n   ⚠️ cited as a spreader in {p.known_case_hits} EUvsDisinfo cases"
            if p.known_case_hits
            else ""
        )
        lines.append(f"<b>{esc(p.name)}</b> — {p.total} flagged\n   {top}{known}\n")
    if report.pairs:
        lines.append(f"{RULE}\n🔗 <b>Channels pushing the same narratives</b>")
        for a, b, n, labels in report.pairs[:6]:
            lines.append(f"  • <b>{esc(a)}</b> ↔ <b>{esc(b)}</b>: {n} shared — {esc('; '.join(labels[:3]))}")
        lines.append("")
    if report.synchronous:
        lines.append(f"{RULE}\n⏱ <b>Synchronous pushes</b> (same cluster, several channels within 24 h)")
        for label, chans, span in report.synchronous[:6]:
            lines.append(f"  • {esc(label)}: {esc(', '.join(chans))} within {_span(span)}")
        lines.append("")
    lines.append(RULE)
    return "\n".join(lines)
