from collections import Counter
from math import log1p

from rank_bm25 import BM25Okapi

from .tokenize import tokenize


class BM25(BM25Okapi):
    def _calc_idf(self, frequencies):
        # Positive IDF for common terms, including a one-passage corpus.
        self.idf = {
            term: log1p((self.corpus_size - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }
        self.average_idf = sum(self.idf.values()) / len(self.idf)


class BM25Index:
    """Tokenize a passage list once and rank many queries against it."""

    def __init__(self, passages: list[str]):
        corpus = [tokenize(passage) for passage in passages]
        self._model = BM25(corpus) if any(corpus) else None

    def rank(self, query: str) -> list[tuple[int, float]]:
        terms = list(Counter(tokenize(query)))
        if not terms or self._model is None:
            return []
        scores = self._model.get_scores(terms)
        return sorted(
            ((i, float(score)) for i, score in enumerate(scores) if score > 0),
            key=lambda pair: -pair[1],
        )


def rank(query: str, passages: list[str]) -> list[tuple[int, float]]:
    return BM25Index(passages).rank(query)
