from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from ..models import Chunk, SearchHit, SearchMode, SearchResult
from ..store.sqlite_store import SQLiteStore
from .bm25 import BM25Index
from .embedder import Embedder, Reranker

_VECTOR_DTYPE = "<f4"
_CACHE_SIZE = 8


def cosine_scores(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    return np.divide(
        vectors @ query, norms, out=np.zeros(len(vectors), dtype=np.float32), where=norms > 0
    )


def rrf(*rankings: list[tuple[int, float]]) -> list[tuple[int, float]]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (index, _) in enumerate(ranking, 1):
            scores[index] += 1 / (60 + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


@dataclass
class _Corpus:
    """Chunks matching one filter, with the BM25 index and vectors built from them."""

    chunks: list[Chunk]
    bm25: BM25Index | None = None
    vectors: dict[str, np.ndarray] = field(default_factory=dict)


class Retriever:
    def __init__(self, store: SQLiteStore, embedder: Embedder | None, reranker: Reranker | None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._cache: OrderedDict[tuple, _Corpus] = OrderedDict()
        self._lock = Lock()

    def _corpus(self, doc_ids: list[str] | None, page: int | None) -> _Corpus:
        # The store generation changes on every ingest or removal, so stale entries never match.
        key = (
            self.store.generation(),
            tuple(sorted(doc_ids)) if doc_ids is not None else None,
            page,
        )
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        corpus = _Corpus(self.store.chunks(doc_ids, page))
        with self._lock:
            self._cache[key] = corpus
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return corpus

    def _lexical(self, query: str, corpus: _Corpus) -> list[tuple[int, float]]:
        if corpus.bm25 is None:
            corpus.bm25 = BM25Index([chunk.text for chunk in corpus.chunks])
        return corpus.bm25.rank(query)

    def _dense(self, query: str, corpus: _Corpus) -> list[tuple[int, float]] | None:
        if self.embedder is None:
            return None
        q = self.embedder.embed([query], query=True)
        if q is None:
            return None
        matrix = corpus.vectors.get(self.embedder.name)
        if matrix is None:
            matrix = self._vectors(corpus.chunks)
            if matrix is None:
                return None
            corpus.vectors[self.embedder.name] = matrix
        scores = cosine_scores(q[0], matrix)
        return sorted(
            ((i, float(score)) for i, score in enumerate(scores) if np.isfinite(score)),
            key=lambda pair: -pair[1],
        )

    def _vectors(self, chunks: list[Chunk]) -> np.ndarray | None:
        """Load passage vectors from the store, embedding and saving any that are missing."""
        ids = [chunk.location.chunk_id for chunk in chunks]
        cached = self.store.embeddings(self.embedder.name, ids)
        missing = [i for i, chunk_id in enumerate(ids) if chunk_id not in cached]
        if missing:
            passages = self.embedder.embed([chunks[i].text for i in missing])
            if passages is None:
                return None
            new = {
                ids[index]: vector.astype(_VECTOR_DTYPE).tobytes()
                for index, vector in zip(missing, passages, strict=True)
            }
            self.store.save_embeddings(self.embedder.name, new)
            cached.update(new)
        return np.stack([np.frombuffer(cached[chunk_id], dtype=_VECTOR_DTYPE) for chunk_id in ids])

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        mode: SearchMode = "hybrid",
        rerank: bool = False,
        *,
        page: int | None = None,
    ) -> SearchResult:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        result = SearchResult(requested_mode=mode, mode=mode)
        corpus = self._corpus(doc_ids, page)
        if not corpus.chunks:
            return result
        dense = self._dense(query, corpus) if mode != "bm25" else None
        if mode != "bm25" and dense is None:
            result.mode = "bm25"
            result.warnings.append("Dense retrieval unavailable. Used BM25.")
        lexical = self._lexical(query, corpus) if result.mode != "dense" else []
        pool = max(top_k, 20) if rerank else top_k
        if result.mode == "hybrid":
            ranking = rrf(lexical[: max(50, pool)], dense[: max(50, pool)])
        else:
            ranking = lexical if result.mode == "bm25" else dense
        hits = [
            SearchHit(**corpus.chunks[index].model_dump(), score=score)
            for index, score in ranking[:pool]
        ]
        if rerank and hits:
            scores = (
                self.reranker.score(query, [hit.text for hit in hits]) if self.reranker else None
            )
            if scores is None:
                result.warnings.append("Reranking unavailable. Used retrieval order.")
            else:
                for hit, score in zip(hits, scores, strict=True):
                    hit.rerank_score = score
                hits.sort(key=lambda hit: -hit.rerank_score)
                result.reranked = True
        result.hits = hits[:top_k]
        return result
