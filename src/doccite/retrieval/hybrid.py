from collections import defaultdict

import numpy as np

from ..models import Chunk, SearchHit, SearchMode, SearchResult
from ..store.sqlite_store import SQLiteStore
from . import bm25
from .embedder import Embedder, Reranker

_VECTOR_DTYPE = "<f4"


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


class Retriever:
    def __init__(self, store: SQLiteStore, embedder: Embedder | None, reranker: Reranker | None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def _dense(self, query: str, chunks: list[Chunk]) -> list[tuple[int, float]] | None:
        if self.embedder is None:
            return None
        q = self.embedder.embed([query], query=True)
        if q is None:
            return None
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
        matrix = np.stack(
            [np.frombuffer(cached[chunk_id], dtype=_VECTOR_DTYPE) for chunk_id in ids]
        )
        scores = cosine_scores(q[0], matrix)
        return sorted(
            ((i, float(score)) for i, score in enumerate(scores) if np.isfinite(score)),
            key=lambda pair: -pair[1],
        )

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
        chunks = self.store.chunks(doc_ids, page)
        if not chunks:
            return result
        texts = [chunk.text for chunk in chunks]
        dense = self._dense(query, chunks) if mode != "bm25" else None
        if mode != "bm25" and dense is None:
            result.mode = "bm25"
            result.warnings.append("Dense retrieval unavailable. Used BM25.")
        lexical = bm25.rank(query, texts) if result.mode != "dense" else []
        pool = max(top_k, 20) if rerank else top_k
        if result.mode == "hybrid":
            ranking = rrf(lexical[: max(50, pool)], dense[: max(50, pool)])
        else:
            ranking = lexical if result.mode == "bm25" else dense
        hits = [
            SearchHit(**chunks[index].model_dump(), score=score) for index, score in ranking[:pool]
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
