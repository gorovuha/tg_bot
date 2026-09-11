"""
scripts/build_index.py — embed the EUvsDisinfo cases, assign top-level narratives, write the retrieval index.

  uv run --group ml python scripts/build_index.py                  # builds data/index/
  uv run --group ml python scripts/build_index.py --query "текст"  # smoke-test: top-5 nearest cases

Input : data/euvsdisinfo/cases.jsonl        (from scripts/fetch_euvsdisinfo.py merge)
        services/narratives.json            (hand-labelled top-level narratives with prototype sentences)
Output: data/index/embeddings.npy           float32 [N, dim], L2-normalised
        data/index/cases.jsonl              same order as embeddings; receipt fields + narrative_id/score
        data/index/narrative_centroids.npy  float32 [K, dim], one unit vector per narrative
        data/index/meta.json                model name, dim, count, narrative ids, built_at

Model: intfloat/multilingual-e5-base (RU/UK/EN and 90+ more; ~280M params, runs on CPU/MPS).
e5 expects "passage: " / "query: " prefixes — keep them in sync with services/retrieval.py.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
CASES = ROOT / "data" / "euvsdisinfo" / "cases.jsonl"
NARRATIVES = ROOT / "services" / "narratives.json"
INDEX = ROOT / "data" / "index"
MODEL = "intfloat/multilingual-e5-base"
KEEP = (
    "id",
    "url",
    "date",
    "title",
    "summary",
    "disproof",
    "outlets",
    "telegram_channels",
    "countries",
    "tags",
)
NARRATIVE_MIN_SIM = 0.80  # below this a case is labelled "other" instead of the nearest narrative


def case_text(c: dict) -> str:
    """What gets embedded: the claim (title) plus its summary — the propaganda side, not the disproof."""
    return f"passage: {c['title']}. {c.get('summary', '')}".strip()


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL)


def load_narratives() -> list[dict]:
    return json.loads(NARRATIVES.read_text())["narratives"]


def assign_narratives(model, emb: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Nearest top-level narrative per case (max cosine over that narrative's prototype sentences)."""
    narratives = load_narratives()
    protos = [f"passage: {p}" for n in narratives for p in n["prototypes"]]
    owner = np.array([i for i, n in enumerate(narratives) for _ in n["prototypes"]])
    proto_emb = model.encode(protos, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    sims = emb @ proto_emb.T
    per_narr = np.full((emb.shape[0], len(narratives)), -1.0, dtype=np.float32)
    for j, o in enumerate(owner):
        per_narr[:, o] = np.maximum(per_narr[:, o], sims[:, j])
    best, score = per_narr.argmax(1), per_narr.max(1)
    ids = [
        narratives[b]["id"] if s >= NARRATIVE_MIN_SIM else "other" for b, s in zip(best, score, strict=True)
    ]
    centroids = np.stack([proto_emb[owner == i].mean(0) for i in range(len(narratives))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return ids, score, centroids


def build(batch_size: int = 64) -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    print(f"{len(cases)} cases from {CASES}", flush=True)
    model = load_model()
    t = time.time()
    emb = model.encode(
        [case_text(c) for c in cases],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"\nembedded in {time.time() - t:.0f}s → {emb.shape}", flush=True)

    narr_ids, narr_score, centroids = assign_narratives(model, emb)
    counts: dict[str, int] = {}
    for n in narr_ids:
        counts[n] = counts.get(n, 0) + 1
    print("narrative assignment:", dict(sorted(counts.items(), key=lambda kv: -kv[1])), flush=True)

    INDEX.mkdir(parents=True, exist_ok=True)
    np.save(INDEX / "embeddings.npy", emb)
    np.save(INDEX / "narrative_centroids.npy", centroids)
    with (INDEX / "cases.jsonl").open("w") as f:
        for c, nid, ns in zip(cases, narr_ids, narr_score, strict=True):
            rec = {k: c.get(k) for k in KEEP}
            rec["narrative_id"], rec["narrative_score"] = nid, round(float(ns), 4)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (INDEX / "meta.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "dim": int(emb.shape[1]),
                "count": int(emb.shape[0]),
                "narratives": [n["id"] for n in load_narratives()],
                "narrative_min_sim": NARRATIVE_MIN_SIM,
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    print(f"index written to {INDEX}", flush=True)


def query(text: str, k: int = 5) -> None:
    emb = np.load(INDEX / "embeddings.npy")
    cases = [json.loads(line) for line in (INDEX / "cases.jsonl").read_text().splitlines()]
    q = load_model().encode([f"query: {text}"], normalize_embeddings=True)[0]
    scores = emb @ q
    for i in np.argsort(-scores)[:k]:
        c = cases[i]
        print(f"{scores[i]:.3f}  {c['date']}  [{c.get('narrative_id')}] {c['title']}\n       {c['url']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="smoke-test the index with a text instead of building")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    if a.query:
        query(a.query)
    else:
        build(a.batch_size)
