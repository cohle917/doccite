import numpy as np

from doccite.retrieval.bm25 import rank
from doccite.retrieval.hybrid import Retriever, rrf
from doccite.retrieval.tokenize import tokenize


class FakeEmbedder:
    name = "test"

    def __init__(self):
        self.document_batches = 0

    def embed(self, texts, *, query=False):
        if not query:
            self.document_batches += 1
        return np.array(
            [
                [1, 0] if "battery" in text.lower() or "recharge" in text.lower() else [0, 1]
                for text in texts
            ],
            dtype=np.float32,
        )


class FakeReranker:
    def score(self, query, texts):
        return [8 if "diesel" in text else -8 for text in texts]


def test_bm25_handles_small_corpus_and_tokenizes_both_languages():
    assert "recoveri" in tokenize("Recoveries")
    assert "深圳" in tokenize("深圳的储能设备")
    assert rank("battery", ["A battery is rechargeable."])[0][1] > 0
    assert rank("missing", ["A battery is rechargeable."]) == []
    assert rank("word", ["..."]) == []


def test_hybrid_dense_cache_rerank_and_filters(tmp_path, indexer, store):
    for name, text in [("a", "The battery stores energy."), ("b", "The diesel motor uses fuel.")]:
        path = tmp_path / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        assert indexer.file(path).status == "indexed"
    embedder = FakeEmbedder()
    retriever = Retriever(store, embedder, FakeReranker())
    assert not retriever.search("recharge", mode="bm25").hits
    assert retriever.search("recharge", mode="dense").hits[0].text.startswith("The battery")
    assert retriever.search("battery").hits[0].text.startswith("The battery")
    assert embedder.document_batches == 1
    assert store.stats().embeddings == 2
    reranked = retriever.search("battery", top_k=1, rerank=True)
    assert reranked.reranked and "diesel" in reranked.hits[0].text
    assert reranked.hits[0].rerank_score == 8
    assert retriever.search("battery", doc_ids=[]).hits == []
    assert rrf([(0, 8), (1, 7)], [(1, 7), (2, 6)])[0][0] == 1
    fallback = Retriever(store, None, None).search("battery", mode="dense", rerank=True)
    assert fallback.mode == "bm25"
    assert len(fallback.warnings) == 2
