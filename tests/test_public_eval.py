import json

import pytest

from doccite.public_eval import claim_metrics, claim_pairs, index_corpus, retrieval_metrics


def test_retrieval_metrics_match_hand_computed_values():
    rankings = [["a", "x", "b"], ["y", "z"]]
    qrels = [{"a": 1, "b": 1}, {"q": 1}]
    metrics = retrieval_metrics(rankings, qrels)
    # Query 1: DCG = 1 + 1/log2(4) = 1.5, IDCG = 1 + 1/log2(3) ≈ 1.631. Query 2 finds nothing.
    assert metrics["nDCG@10"] == pytest.approx((1.5 / (1 + 1 / 1.5849625)) / 2)
    assert metrics["Recall@100"] == pytest.approx(0.5)
    assert metrics["MRR@10"] == pytest.approx(0.5)


def test_claim_pairs_and_metrics(tmp_path):
    path = tmp_path / "claims.jsonl"
    rows = [
        {"claim": "A", "evidence": {"1": [{"label": "SUPPORT"}]}, "cited_doc_ids": [1, 2]},
        {"claim": "B", "evidence": {"3": [{"label": "CONTRADICT"}]}, "cited_doc_ids": [3]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    pairs = claim_pairs(path)
    assert pairs == [("A", "1", "SUPPORT"), ("A", "2", "NEI"), ("B", "3", "CONTRADICT")]
    metrics = claim_metrics([label for *_, label in pairs], ["SUPPORT", "SUPPORT", "CONTRADICT"])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["false_support"] == (1, 2)
    assert metrics["f1"]["NEI"] == 0


def test_index_corpus_is_incremental(tmp_path, settings, store):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "7", "title": "Ferry", "text": "Costs fell by 12%."}), encoding="utf-8"
    )
    ids = index_corpus(settings, store, corpus)
    generation = store.generation()
    assert index_corpus(settings, store, corpus) == ids
    assert store.generation() == generation
    [chunk] = store.chunks([ids["7"]])
    assert chunk.text == "Ferry\n\nCosts fell by 12%."
