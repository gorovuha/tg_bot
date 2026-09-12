"""
services/retrieval.py — classifier backend: nearest EUvsDisinfo cases by multilingual embeddings.

Given a message, embed it (multilingual-e5-base, "query: " prefix), find the top-k most similar
cases in data/index/, and decide "propaganda" when the best similarity clears a calibrated threshold.
Receipts are the matching cases (title, link, disproof excerpt). The narrative label is the
top-level narrative (services/narratives.json) of the strongest matches.

Threshold/calibration come from data/index/calibration.json (written by scripts/evaluate.py);
RETRIEVAL_THRESHOLD in the environment overrides the threshold.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from services.classifier import ClassificationResult, Receipt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "data" / "index"
NARRATIVES_PATH = ROOT / "services" / "narratives.json"
BACKEND = "retrieval"
TOP_K = 5
MAX_RECEIPTS = 3
# Fallback calibration before scripts/evaluate.py has run (cosine → probability, threshold).
DEFAULT_CALIBRATION = {"threshold": 0.825, "a": 150.0, "b": -123.75}  # b = -a * threshold

Encoder = Callable[[list[str]], np.ndarray]


class IndexMissingError(RuntimeError):
    pass


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RetrievalClassifier:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        encoder: Encoder | None = None,
        narratives_path: Path = NARRATIVES_PATH,
    ) -> None:
        if not (index_dir / "embeddings.npy").exists():
            raise IndexMissingError(
                f"No retrieval index at {index_dir}. Build it: "
                "python scripts/fetch_euvsdisinfo.py merge && python scripts/build_index.py"
            )
        self.index_dir = index_dir
        self.emb: np.ndarray = np.load(index_dir / "embeddings.npy")
        self.cases: list[dict] = [
            json.loads(line) for line in (index_dir / "cases.jsonl").read_text().splitlines()
        ]
        self.meta: dict = json.loads((index_dir / "meta.json").read_text())
        self.narratives: list[dict] = json.loads(narratives_path.read_text())["narratives"]
        self.narrative_label = {n["id"]: n["label"] for n in self.narratives}
        self._proto_emb: np.ndarray | None = None  # prototype sentence vectors, encoded lazily
        self._proto_owner: np.ndarray | None = None
        self.calibration = dict(DEFAULT_CALIBRATION)
        calib_path = index_dir / "calibration.json"
        if calib_path.exists():
            self.calibration.update(json.loads(calib_path.read_text()))
        env_thr = os.getenv("RETRIEVAL_THRESHOLD")
        if env_thr:
            self.calibration["threshold"] = float(env_thr)
        self._encoder = encoder
        logger.info(
            "Retrieval index: %d cases, model %s, threshold %.3f",
            len(self.cases),
            self.meta.get("model"),
            self.calibration["threshold"],
        )

    # ── encoding ──────────────────────────────────────────────────────────
    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.meta["model"])
            self._encoder = lambda t: model.encode(
                t, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
        return np.asarray(self._encoder(texts), dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])[0]

    def _direct_narrative_scores(self, q: np.ndarray) -> dict[str, float]:
        """Similarity of the query to each narrative's prototype sentences (max over prototypes)."""
        if self._proto_emb is None:
            protos = [f"passage: {p}" for n in self.narratives for p in n["prototypes"]]
            self._proto_owner = np.array([i for i, n in enumerate(self.narratives) for _ in n["prototypes"]])
            self._proto_emb = self._encode(protos) if protos else np.zeros((0, q.shape[0]), dtype=np.float32)
        if not len(self._proto_emb):
            return {}
        sims = self._proto_emb @ q
        out: dict[str, float] = {}
        for j, o in enumerate(self._proto_owner):
            nid = self.narratives[o]["id"]
            out[nid] = max(out.get(nid, -1.0), float(sims[j]))
        return out

    # ── classification ────────────────────────────────────────────────────
    def confidence(self, score: float) -> float:
        return round(_sigmoid(self.calibration["a"] * score + self.calibration["b"]), 3)

    def classify_vector(self, q: np.ndarray, top_k: int = TOP_K) -> ClassificationResult:
        sims = self.emb @ q
        top = np.argsort(-sims)[:top_k]
        best = float(sims[top[0]])
        thr = self.calibration["threshold"]
        is_prop = best >= thr

        # Narrative = matching cases' narratives (vote weighted by similarity, top-1 counted twice)
        # blended with the query's own similarity to each narrative's prototype sentences.
        votes: dict[str, float] = {}
        for rank, i in enumerate(top):
            if sims[i] >= thr or rank == 0:
                nid = self.cases[i].get("narrative_id") or "other"
                votes[nid] = votes.get(nid, 0.0) + float(sims[i]) * (2.0 if rank == 0 else 1.0)
        direct = self._direct_narrative_scores(q)
        if direct:
            # Prototype similarity decides; case votes (share of vote mass, x0.05) only break near-ties.
            # Per-case narrative_id in the index is noisy (nearest prototype over title+summary), so it
            # must not outweigh the query's own match against the prototypes.
            total = sum(votes.values()) or 1.0
            scores = {nid: direct[nid] + 0.05 * votes.get(nid, 0.0) / total for nid in direct}
            narrative_id = max(scores, key=scores.__getitem__)
        else:
            narrative_id = max(votes, key=votes.__getitem__)
        if narrative_id == "other":  # fall back to the title of the best case
            label = self.cases[top[0]]["title"]
        else:
            label = self.narrative_label.get(narrative_id, narrative_id)

        evidence = [
            Receipt(
                title=self.cases[i]["title"],
                url=self.cases[i].get("url"),
                similarity=float(sims[i]),
                excerpt=(self.cases[i].get("disproof") or "")[:300] or None,
            )
            for i in top[:MAX_RECEIPTS]
        ]
        return ClassificationResult(
            is_propaganda=is_prop,
            confidence=self.confidence(best) if is_prop else 1 - self.confidence(best),
            narrative_label=label if is_prop else "None detected",
            cluster_id=narrative_id if is_prop else None,
            backend=BACKEND,
            evidence=evidence if is_prop else evidence[:1],
            embedding=q.tolist(),
        )

    def classify(self, text: str) -> ClassificationResult:
        return self.classify_vector(self.embed(text))


def known_telegram_channels(cases: list[dict]) -> dict[str, int]:
    """Telegram channel usernames (lowercase) that EUvsDisinfo cases cite as spreaders → number of cases."""
    counts: dict[str, int] = {}
    for c in cases:
        for u in c.get("telegram_channels") or []:
            counts[u.lower()] = counts.get(u.lower(), 0) + 1
    return counts


_instance: RetrievalClassifier | None = None


def get_classifier() -> RetrievalClassifier:
    global _instance
    if _instance is None:
        _instance = RetrievalClassifier()
    return _instance
