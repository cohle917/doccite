"""Local natural language inference with an ONNX cross-encoder (no API key)."""

import json
import logging
from os import cpu_count
from threading import Lock
from typing import Protocol

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)

LABELS = ("entailment", "neutral", "contradiction")


class NLIModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]] | None:
        """Return label probabilities for (premise, hypothesis) pairs, or None if unavailable."""
        ...


class OnnxNLI:
    def __init__(self, settings: Settings, *, batch_size: int = 16, max_length: int = 320):
        self.name = settings.nli_model
        self.onnx_file = settings.nli_onnx_file
        self.cache_dir = settings.data_dir / "models"
        self.batch_size = batch_size
        self.max_length = max_length
        self._session = None
        self._failed = False
        self._lock = Lock()

    def _load(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        def fetch(filename: str) -> str:
            return hf_hub_download(self.name, filename, cache_dir=str(self.cache_dir))

        config = json.loads(open(fetch("config.json"), encoding="utf-8").read())
        index = {label.lower(): int(i) for i, label in config["id2label"].items()}
        if not set(LABELS) <= set(index):
            raise ValueError(f"{self.name} does not have NLI labels: {sorted(index)}")
        self._label_index = [index[label] for label in LABELS]
        self._tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
        self._tokenizer.enable_truncation(self.max_length)
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(8, cpu_count() or 1))
        self._session = ort.InferenceSession(
            fetch(self.onnx_file), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {item.name for item in self._session.get_inputs()}

    def _batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(pairs)
        width = max(len(encoding.ids) for encoding in encodings)
        feed = {
            "input_ids": np.zeros((len(pairs), width), dtype=np.int64),
            "attention_mask": np.zeros((len(pairs), width), dtype=np.int64),
            "token_type_ids": np.zeros((len(pairs), width), dtype=np.int64),
        }
        pad = self._tokenizer.token_to_id("[PAD]") or 0
        feed["input_ids"].fill(pad)
        for row, encoding in enumerate(encodings):
            size = len(encoding.ids)
            feed["input_ids"][row, :size] = encoding.ids
            feed["attention_mask"][row, :size] = encoding.attention_mask
            feed["token_type_ids"][row, :size] = encoding.type_ids
        logits = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
        logits = logits[:, self._label_index]
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]] | None:
        if not pairs:
            return []
        with self._lock:
            if self._failed:
                return None
            try:
                if self._session is None:
                    self._load()
                # Sort by length so each batch pads to a similar width.
                order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
                results: list[dict[str, float] | None] = [None] * len(pairs)
                for start in range(0, len(order), self.batch_size):
                    chunk = order[start : start + self.batch_size]
                    probabilities = self._batch([pairs[i] for i in chunk])
                    for i, row in zip(chunk, probabilities, strict=True):
                        results[i] = dict(zip(LABELS, map(float, row), strict=True))
                return results
            except Exception as exc:
                self._failed = True
                logger.warning("NLI model unavailable (%s): %s", self.name, exc)
                return None
