"""
services/classifier.py
Classification layer: `classify(text)` returns a ClassificationResult.

Backends (selected with CLASSIFIER_BACKEND, default "retrieval"):
  retrieval  — embedding search over EUvsDisinfo cases (services/retrieval.py); needs data/index/.
  mock       — deterministic keyword heuristics (RU/UK/EN). Demo/tests only; never a real model.

`ClassificationResult` is the contract between the bot and any backend. Extend it,
don't leak backend internals into handlers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MOCK_BACKEND = "mock"
RETRIEVAL_BACKEND = "retrieval"


@dataclass
class Receipt:
    """One piece of evidence for a flag (e.g. a matching EUvsDisinfo case)."""

    title: str
    url: str | None = None
    similarity: float = 0.0
    excerpt: str | None = None


@dataclass
class ClassificationResult:
    is_propaganda: bool
    confidence: float
    narrative_label: str
    cluster_id: str | None = None
    backend: str = MOCK_BACKEND
    evidence: list[Receipt] = field(default_factory=list)
    embedding: list[float] | None = None  # query vector (retrieval backend) — stored for clustering

    def __str__(self) -> str:
        flag = "PROPAGANDA" if self.is_propaganda else "CLEAN"
        return f"{flag} [{self.confidence:.0%}] {self.narrative_label} ({self.backend})"


def backend_name() -> str:
    return os.getenv("CLASSIFIER_BACKEND", RETRIEVAL_BACKEND).strip().lower() or RETRIEVAL_BACKEND


def ensure_backend_ready() -> None:
    """Fail fast at startup if the configured backend can't run (no silent fallback to mock)."""
    name = backend_name()
    if name == RETRIEVAL_BACKEND:
        from services.retrieval import get_classifier

        get_classifier()  # raises IndexMissingError with build instructions
    elif name != MOCK_BACKEND:
        raise RuntimeError(f"Unknown CLASSIFIER_BACKEND={name!r} (use 'retrieval' or 'mock')")


async def classify(text: str) -> ClassificationResult:
    """Classify `text` with the configured backend."""
    name = backend_name()
    if name == MOCK_BACKEND:
        return mock_classify(text)
    from services.retrieval import get_classifier

    clf = get_classifier()
    return await asyncio.to_thread(clf.classify, text)  # embedding is CPU-bound; keep the event loop free


# ── Mock classifier ──────────────────────────────────────────────────────────
# Deterministic keyword heuristics. Word-boundary regexes so "nato" no longer
# matches "senator". Same text always yields the same result.

_SIGNALS: list[tuple[str, str, str, float]] = [
    # (regex, narrative_label, cluster_id, weight)
    (r"\bnato\b|\bнато\b", "Anti-NATO destabilisation", "cluster_nato", 1.0),
    (
        r"\bbiolabs?\b|\bbioweapons?\b|\bбиолаборатори\w*|\bбиооруж\w*|\bбіолаборатор\w*",
        "Biolabs conspiracy",
        "cluster_biolabs",
        1.5,
    ),
    (r"\bdeep state\b|\bглубинн\w+ государств\w*", "Deep-state narrative", "cluster_deep_state", 1.0),
    (
        r"\bzelensky\b|\bzelenskyy\b|\bзеленск\w*|\bзеленськ\w*",
        "Zelensky delegitimisation",
        "cluster_zelensky",
        0.7,
    ),
    (r"\bfalse flag\b|\bложн\w+ флаг\w*|\bпровокаци\w+", "False-flag accusation", "cluster_false_flag", 1.0),
    (r"\bgenocide\b|\bгеноцид\w*", "Genocide framing", "cluster_genocide", 1.0),
    (
        r"\bnazis?\b|\bneo-nazis?\b|\bнацист\w*|\bнаци\b|\bнацик\w*|\bбандеров\w*",
        "Neo-Nazi labelling",
        "cluster_nazi",
        1.2,
    ),
    (
        r"\bspecial (military )?operation\b|\bспецоперац\w*|\bсво\b",
        "War-euphemism framing",
        "cluster_special_op",
        1.0,
    ),
    (r"\bukraine is losing\b|\bукраина проигр\w*", "Defeatism narrative", "cluster_defeatism", 1.0),
    (r"\bwestern media lies?\b|\bзападн\w+ сми (лгут|врут)\b", "MSM distrust frame", "cluster_msm", 1.0),
    (r"\bpuppet\b|\bмарионетк\w*", "Puppet-regime framing", "cluster_puppet", 0.8),
    (
        r"\bкиевск\w+ режим\w*|\bkiev regime\b|\bkyiv regime\b",
        "Regime delegitimisation",
        "cluster_regime",
        1.2,
    ),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), label, cid, w) for p, label, cid, w in _SIGNALS]


def mock_classify(text: str) -> ClassificationResult:
    hits = [(label, cid, w) for rx, label, cid, w in _COMPILED if rx.search(text)]
    if not hits:
        return ClassificationResult(False, 0.9, "None detected", None, MOCK_BACKEND)

    # Strongest signal decides the label; more/heavier hits raise confidence.
    label, cid, _ = max(hits, key=lambda h: h[2])
    total = sum(w for _, _, w in hits)
    confidence = round(min(0.95, 0.55 + 0.15 * total), 2)
    return ClassificationResult(True, confidence, label, cid, MOCK_BACKEND)
