import json

import numpy as np
import pytest

from services.retrieval import IndexMissingError, RetrievalClassifier

DIM = 4


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index(tmp_path):
    """Tiny index: case 0/1 are 'biolabs', case 2 is 'crimea', case 3 unlabelled."""
    emb = np.stack([_unit([1, 0, 0, 0]), _unit([0.9, 0.1, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])])
    np.save(tmp_path / "embeddings.npy", emb)
    cases = [
        {
            "id": "a",
            "title": "US biolabs in Ukraine",
            "url": "https://x/a",
            "disproof": "No.",
            "narrative_id": "biolabs",
        },
        {
            "id": "b",
            "title": "Pentagon labs",
            "url": "https://x/b",
            "disproof": "Nope.",
            "narrative_id": "biolabs",
        },
        {
            "id": "c",
            "title": "Crimea referendum",
            "url": "https://x/c",
            "disproof": "Illegal.",
            "narrative_id": "crimea_legal",
        },
        {"id": "d", "title": "Something odd", "url": None, "disproof": "", "narrative_id": "other"},
    ]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
    (tmp_path / "meta.json").write_text(json.dumps({"model": "stub", "dim": DIM, "count": 4}))
    (tmp_path / "calibration.json").write_text(json.dumps({"threshold": 0.8, "a": 60.0, "b": -49.2}))
    return tmp_path


@pytest.fixture
def narratives(tmp_path):
    p = tmp_path / "narratives.json"
    p.write_text(
        json.dumps(
            {
                "narratives": [
                    {"id": "biolabs", "label": "US biolabs develop bioweapons", "prototypes": []},
                    {"id": "crimea_legal", "label": "Crimea legally joined Russia", "prototypes": []},
                ]
            }
        )
    )
    return p


def make(index, narratives, vec):
    return RetrievalClassifier(
        index_dir=index, narratives_path=narratives, encoder=lambda texts: np.stack([_unit(vec)])
    )


def test_missing_index_raises(tmp_path):
    with pytest.raises(IndexMissingError):
        RetrievalClassifier(index_dir=tmp_path / "nope")


def test_propaganda_with_receipts_and_narrative(index, narratives):
    clf = make(index, narratives, [1, 0.05, 0, 0])
    r = clf.classify("биолаборатории")
    assert r.is_propaganda and r.backend == "retrieval"
    assert r.narrative_label == "US biolabs develop bioweapons" and r.cluster_id == "biolabs"
    assert [e.title for e in r.evidence][:2] == ["US biolabs in Ukraine", "Pentagon labs"]
    assert r.evidence[0].url == "https://x/a" and r.evidence[0].excerpt == "No."
    assert 0.5 < r.confidence <= 1.0 and len(r.embedding) == DIM


def test_clean_below_threshold(index, narratives):
    clf = make(index, narratives, [0.5, 0.5, 0.5, 0.5])  # cos ≈ 0.5 to everything
    r = clf.classify("weather")
    assert not r.is_propaganda and r.narrative_label == "None detected" and r.cluster_id is None
    assert len(r.evidence) == 1  # nearest case still shown for transparency


def test_other_narrative_falls_back_to_case_title(index, narratives):
    r = make(index, narratives, [0, 0, 1, 0]).classify("x")
    assert r.is_propaganda and r.narrative_label == "Something odd" and r.cluster_id == "other"


def test_env_threshold_override(index, narratives, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.9999")
    r = make(index, narratives, [1, 0.05, 0, 0]).classify("x")
    assert not r.is_propaganda
