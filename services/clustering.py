"""
services/clustering.py — narrative clusters over time and channel comparison.

Works on flagged messages (storage.db rows joined with messages: text, timestamp, embedding, origin).
No model calls here: embeddings were stored at classification time.

- cluster_messages(rows)  → list[Cluster]: hard partition by top-level narrative (cluster_id), then
  agglomerative clustering on message embeddings inside each narrative (cosine distance threshold);
  rows without embeddings stay grouped by narrative only.
- channel_report(rows)    → ChannelReport: per-channel narrative profile, channel pairs that share
  narratives, and clusters that appeared in several channels within a short window ("synchronous").
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations

import numpy as np

CLUSTER_DISTANCE = 0.16  # cosine distance; messages closer than this join one cluster (sim ≥ 0.84)
SYNC_WINDOW = timedelta(hours=24)
SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class Cluster:
    id: int
    label: str  # majority narrative label
    narrative_id: str | None
    rows: list = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def avg_confidence(self) -> float:
        return sum(r["confidence"] for r in self.rows) / len(self.rows)

    @property
    def sources(self) -> Counter:
        return Counter(source_name(r) for r in self.rows)

    def daily_counts(self, days: int = 14, today: date | None = None) -> list[int]:
        today = today or datetime.now().astimezone().date()
        start = today - timedelta(days=days - 1)
        counts = [0] * days
        for r in self.rows:
            d = _ts(r).date()
            if start <= d <= today:
                counts[(d - start).days] += 1
        return counts


@dataclass
class ChannelProfile:
    name: str
    total: int
    narratives: Counter  # narrative_id/label → count
    known_case_hits: int = 0  # how many EUvsDisinfo cases list this channel as a spreader


@dataclass
class ChannelReport:
    channels: list[ChannelProfile]
    pairs: list[tuple[str, str, int, list[str]]]  # (a, b, shared narrative count, shared narrative labels)
    synchronous: list[tuple[str, list[str], timedelta]]  # (cluster label, channels, span)


# ── helpers ──────────────────────────────────────────────────────────────────


def _ts(row: Mapping) -> datetime:
    dt = datetime.fromisoformat(row["timestamp"])
    return dt.astimezone() if dt.tzinfo else dt


def _get(row: Mapping, key: str):
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def source_name(row: Mapping) -> str:
    """Where the message originates: forwarded-from channel if any, else the posting channel/chat."""
    return (
        _get(row, "fwd_chat_title")
        or _get(row, "fwd_chat_username")
        or _get(row, "sender_chat_title")
        or _get(row, "chat_title")
        or "unknown"
    )


def source_username(row: Mapping) -> str | None:
    u = _get(row, "fwd_chat_username")
    return u.lower() if u else None


def decode_embedding(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def sparkline(counts: Sequence[int]) -> str:
    top = max(counts) or 1
    return "".join(
        SPARK[min(len(SPARK) - 1, round(c / top * (len(SPARK) - 1)))] if c else SPARK[0] for c in counts
    )


# ── clustering ───────────────────────────────────────────────────────────────


def _labels_by_embedding(vectors: np.ndarray, distance: float) -> np.ndarray:
    if len(vectors) == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance
    )
    return model.fit_predict(vectors)


def cluster_messages(rows: Sequence[Mapping], distance: float = CLUSTER_DISTANCE) -> list[Cluster]:
    """Group flagged messages into narrative clusters; largest first."""
    rows = list(rows)
    if not rows:
        return []
    # Two levels: the top-level narrative is a hard partition (keeps labels clean), then embeddings
    # split each narrative into concrete stories. Rows without embeddings (mock backend) stay one group.
    by_narrative: dict[str, list] = defaultdict(list)
    for r in rows:
        by_narrative[str(_get(r, "cluster_id") or _get(r, "narrative_label"))].append(r)
    groups: dict[str, list] = defaultdict(list)
    for nid, members in by_narrative.items():
        vectors = [decode_embedding(_get(r, "embedding")) for r in members]
        if all(v is not None for v in vectors):
            labels = _labels_by_embedding(np.stack(vectors), distance)
            for r, lab in zip(members, labels, strict=True):
                groups[f"{nid}:{lab}"].append(r)
        else:
            groups[nid] = members

    clusters: list[Cluster] = []
    for members in groups.values():
        labels = Counter(m["narrative_label"] for m in members)
        nids = Counter(_get(m, "cluster_id") for m in members if _get(m, "cluster_id"))
        times = [_ts(m) for m in members]
        clusters.append(
            Cluster(
                id=0,
                label=labels.most_common(1)[0][0],
                narrative_id=nids.most_common(1)[0][0] if nids else None,
                rows=sorted(members, key=_ts, reverse=True),
                first_seen=min(times),
                last_seen=max(times),
            )
        )
    clusters.sort(key=lambda c: (-c.size, c.last_seen and -c.last_seen.timestamp()))
    for i, c in enumerate(clusters, 1):
        c.id = i
    return clusters


# ── channels ─────────────────────────────────────────────────────────────────


def _tightest_window(posts: list[tuple[datetime, str]], n_channels: int) -> timedelta | None:
    """Smallest time span that contains at least one post from every channel (sliding window)."""
    posts = sorted(posts)
    best: timedelta | None = None
    seen: Counter = Counter()
    left = 0
    for right, (t_right, chan) in enumerate(posts):
        seen[chan] += 1
        while len(seen) == n_channels:
            span = t_right - posts[left][0]
            best = span if best is None or span < best else best
            seen[posts[left][1]] -= 1
            if seen[posts[left][1]] == 0:
                del seen[posts[left][1]]
            left += 1
    return best


def channel_report(
    rows: Sequence[Mapping],
    known_channels: Mapping[str, int] | None = None,
    clusters: list[Cluster] | None = None,
    min_shared: int = 1,
) -> ChannelReport:
    rows = list(rows)
    known_channels = known_channels or {}
    by_channel: dict[str, list] = defaultdict(list)
    for r in rows:
        by_channel[source_name(r)].append(r)

    profiles: list[ChannelProfile] = []
    for name, members in by_channel.items():
        narr = Counter((_get(m, "cluster_id") or m["narrative_label"]) for m in members)
        usernames = {source_username(m) for m in members} - {None}
        hits = sum(known_channels.get(u, 0) for u in usernames)
        profiles.append(ChannelProfile(name=name, total=len(members), narratives=narr, known_case_hits=hits))
    profiles.sort(key=lambda p: -p.total)

    pairs: list[tuple[str, str, int, list[str]]] = []
    label_of: dict[str, str] = {}
    for r in rows:
        label_of.setdefault(_get(r, "cluster_id") or r["narrative_label"], r["narrative_label"])
    for a, b in combinations(profiles, 2):
        shared = set(a.narratives) & set(b.narratives)
        if len(shared) >= min_shared:
            pairs.append((a.name, b.name, len(shared), [label_of[s] for s in sorted(shared)]))
    pairs.sort(key=lambda p: -p[2])

    synchronous: list[tuple[str, list[str], timedelta]] = []
    for c in clusters or cluster_messages(rows):
        chans = c.sources
        if len(chans) >= 2:
            span = _tightest_window([(_ts(r), source_name(r)) for r in c.rows], len(chans))
            if span is not None and span <= SYNC_WINDOW:
                synchronous.append((c.label, sorted(chans), span))
    synchronous.sort(key=lambda s: (-len(s[1]), s[2]))
    return ChannelReport(channels=profiles, pairs=pairs, synchronous=synchronous)
