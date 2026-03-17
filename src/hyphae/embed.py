"""Embedding wrapper — lazy-loads sentence-transformers."""
from __future__ import annotations

import numpy as np


class Embedder:
    """Thin wrapper around sentence-transformers. Lazy-loaded."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.model  # trigger lazy load
        return self._dim  # type: ignore

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized float32 embeddings."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]
