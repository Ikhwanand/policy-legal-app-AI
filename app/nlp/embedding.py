from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class VectorIndex:
    def __init__(self, dim: int, storage_dir: str):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "faiss.index")
        self.meta_path = os.path.join(storage_dir, "meta.pkl")
        self.model = _load_sentence_transformer(MODEL_NAME)
        self.index: faiss.Index | None = None
        self.meta: List[dict] = []
        self._dim = dim

    def _ensure(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        elif getattr(self.index, "d", dim) != dim:
            raise ValueError(f"Index dimension mismatch: expected {dim}, found {self.index.d}")

    def is_empty(self) -> bool:
        return self.index is None or getattr(self.index, "ntotal", 0) == 0

    def add_texts(self, texts: List[str], meta_list: List[dict]):
        if not texts:
            return
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._ensure(embs.shape[1])
        self.index.add(embs.astype(np.float32))
        start = len(self.meta)
        for offset, (text, meta) in enumerate(zip(texts, meta_list)):
            stored = dict(meta)
            stored.setdefault("text", text)
            stored.setdefault("vector_id", start + offset)
            self.meta.append(stored)

    def search(self, query: str, k: int = 5) -> List[Tuple[float, dict]]:
        if self.is_empty():
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q, k)
        results: List[Tuple[float, dict]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(self.meta):
                continue
            results.append((float(score), self.meta[idx]))
        return results

    def save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.meta = pickle.load(f) or []
        else:
            self.meta = []
        if self.index is not None and getattr(self.index, "d", self._dim) != self._dim:
            raise ValueError("Dimensi index tidak sesuai dengan model embedding.")