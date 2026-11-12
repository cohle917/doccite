import logging
from threading import Lock
from typing import Protocol

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray | None: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float] | None: ...


class FastEmbedder:
    def __init__(self, settings: Settings):
        self.name = settings.embedding_model
        self.cache_dir = settings.data_dir / "models"
        self._model = None
        self._failed = False
        self._lock = Lock()

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray | None:
        with self._lock:
            if self._failed:
                return None
            try:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(self.name, cache_dir=str(self.cache_dir), threads=2)
                vectors = (
                    self._model.query_embed(texts) if query else self._model.passage_embed(texts)
                )
                return np.asarray(list(vectors), dtype=np.float32)
            except Exception as exc:
                self._failed = True
                logger.warning("Embedding unavailable (%s): %s", self.name, exc)
                return None


class FastReranker:
    def __init__(self, settings: Settings):
        self.name = settings.reranker_model
        self.cache_dir = settings.data_dir / "models"
        self._model = None
        self._failed = False
        self._lock = Lock()

    def score(self, query: str, texts: list[str]) -> list[float] | None:
        with self._lock:
            if self._failed:
                return None
            try:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(
                        self.name, cache_dir=str(self.cache_dir), threads=2
                    )
                return [float(score) for score in self._model.rerank(query, texts)]
            except Exception as exc:
                self._failed = True
                logger.warning("Reranker unavailable (%s): %s", self.name, exc)
                return None
