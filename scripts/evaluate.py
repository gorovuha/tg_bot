"""
scripts/evaluate.py — measure the retrieval classifier and calibrate its threshold.

  uv run --group ml python scripts/evaluate.py [--pos 1500] [--neg 1500] [--seed 0]

Positives: held-out EUvsDisinfo cases. The query is the case *summary* (the claim as it circulates);
           the case's own index row is masked so it cannot match itself.
Negatives: neutral news — AG News (EN), Gazeta (RU), ukrainian-news (UK) — sampled via HF datasets.
Score    : cosine similarity to the nearest remaining case (top-1), exactly what the bot uses.

Outputs  : data/index/calibration.json  {threshold, a, b}  (sigmoid(a*score+b) ≈ P(propaganda))
           prints AUC, the precision/recall table and narrative-agreement on positives.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.retrieval import INDEX_DIR, RetrievalClassifier

MIN_PRECISION = 0.93

NEGATIVE_SOURCES = [
    # (dataset, config, split, text field, language)
    ("fancyzhx/ag_news", None, "test", "text", "en"),
    ("IlyaGusev/gazeta", None, "test", "summary", "ru"),
    (
        "wikimedia/wikipedia",
        "20231101.uk",
        "train",
        "text",
        "uk",
    ),  # no maintained UK news set; encyclopedic text
]


def load_negatives(n_per_source: int, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    out: list[tuple[str, str]] = []
    for name, config, split, field, lang in NEGATIVE_SOURCES:
        try:
            ds = load_dataset(name, config, split=split, streaming=True).shuffle(seed=seed, buffer_size=5000)
            texts = [str(r[field]).strip()[:600] for r in itertools.islice(ds, n_per_source) if r.get(field)]
            out += [(t, lang) for t in texts if len(t) > 30]
            print(f"  negatives {lang}: {len(texts)} from {name}")
        except Exception as exc:
            print(f"  WARN could not load {name}: {exc}")
    return out


def metrics_at(scores: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(
    scores: np.ndarray, y: np.ndarray, iters: int = 3000, lr: float = 0.5
) -> tuple[float, float]:
    """1-D logistic regression by gradient descent on standardised scores (no sklearn dependency needed)."""
    mu, sd = scores.mean(), scores.std() + 1e-9
    z = (scores - mu) / sd
    w, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(w * z + b)))
        g = p - y
        w -= lr * float((g * z).mean())
        b -= lr * float(g.mean())
    return w / sd, b - w * mu / sd  # back to raw-score coefficients


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=1500)
    ap.add_argument("--neg", type=int, default=600, help="per negative source")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    clf = RetrievalClassifier()
    with_summary = [i for i, c in enumerate(clf.cases) if len(c.get("summary") or "") > 40]
    pos_idx = rng.choice(with_summary, size=min(a.pos, len(with_summary)), replace=False)
    print(f"positives: {len(pos_idx)} held-out cases (query = summary, own row masked)")
    negatives = load_negatives(a.neg, a.seed)
    print(f"negatives: {len(negatives)}")

    pos_texts = [clf.cases[i]["summary"][:600] for i in pos_idx]
    neg_texts = [t for t, _ in negatives]
    q = clf._encode([f"query: {t}" for t in pos_texts + neg_texts])
    sims = q @ clf.emb.T
    for row, i in enumerate(pos_idx):
        sims[row, i] = -1.0  # mask self
    top1 = sims.max(1)
    top_idx = sims.argmax(1)
    y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
    scores = top1

    print(f"\nAUC = {auc(scores, y):.4f}")
    print(f"positives: median {np.median(scores[y == 1]):.3f}  p10 {np.percentile(scores[y == 1], 10):.3f}")
    for lang in ("en", "ru", "uk"):
        m = np.array([False] * len(pos_texts) + [lg == lang for _, lg in negatives])
        if m.any():
            print(
                f"negatives {lang}: median {np.median(scores[m]):.3f}  p90 {np.percentile(scores[m], 90):.3f}"
                f"  p99 {np.percentile(scores[m], 99):.3f}"
            )

    print("\nthreshold  precision  recall  F1")
    best_f1, best_thr = 0.0, 0.0
    for thr in np.arange(0.78, 0.90, 0.005):
        p, r, f = metrics_at(scores, y, thr)
        print(f"  {thr:.3f}     {p:.3f}     {r:.3f}   {f:.3f}")
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    # Watchdog bias: prefer precision, but short RU/UK slogans score ~0.82-0.83, so don't overshoot.
    # Pick the smallest threshold with precision >= MIN_PRECISION (recall >= 0.5), else the F1-optimal one.
    chosen = best_thr
    for thr in np.arange(0.78, 0.90, 0.001):
        p, r, _ = metrics_at(scores, y, thr)
        if p >= MIN_PRECISION and r >= 0.5:
            chosen = float(thr)
            break
    p, r, f = metrics_at(scores, y, chosen)
    print(
        f"\nchosen threshold {chosen:.3f}: precision {p:.3f} recall {r:.3f} F1 {f:.3f} (F1-optimal was {best_thr:.3f})"
    )

    # narrative agreement on positives that are flagged
    agree = tot = 0
    for row, i in enumerate(pos_idx):
        own = clf.cases[i].get("narrative_id")
        if scores[row] >= chosen and own and own != "other":
            tot += 1
            agree += clf.cases[top_idx[row]].get("narrative_id") == own
    print(
        f"narrative agreement (top-1 case shares narrative with held-out case): {agree}/{tot} = {agree / max(tot, 1):.3f}"
    )

    w, _ = fit_logistic(scores, y)
    b = -w * chosen  # centre the sigmoid on the decision threshold so a flagged message never shows < 50%
    calib = {
        "threshold": round(chosen, 4),
        "a": round(float(w), 3),
        "b": round(float(b), 3),
        "auc": round(auc(scores, y), 4),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "n_pos": len(pos_texts),
        "n_neg": len(neg_texts),
    }
    (INDEX_DIR / "calibration.json").write_text(json.dumps(calib, indent=2))
    print(f"\nwrote {INDEX_DIR / 'calibration.json'}: {calib}")


if __name__ == "__main__":
    main()
