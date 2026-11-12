from collections import Counter

from rank_bm25 import BM25Okapi

from .tokenize import tokenize


class BM25(BM25Okapi):
    def _calc_idf(self, frequencies):
        # Positive IDF for common terms, including a one-passage corpus.
        from math import log1p

        self.idf = {
            term: log1p((self.corpus_size - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }
        self.average_idf = sum(self.idf.values()) / len(self.idf)


def rank(query: str, passages: list[str]) -> list[tuple[int, float]]:
    terms = list(Counter(tokenize(query)))
    corpus = [tokenize(passage) for passage in passages]
    if not terms or not any(corpus):
        return []
    scores = BM25(corpus).get_scores(terms)
    return sorted(
        ((i, float(score)) for i, score in enumerate(scores) if score > 0),
        key=lambda pair: -pair[1],
    )
