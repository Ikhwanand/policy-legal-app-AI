import logging
from pathlib import Path
from threading import RLock
from typing import List, Dict

from app.nlp.embedding import VectorIndex
from app.nlp.ingest import build_chunks

logger = logging.getLogger(__name__)


class KnowledgeStore:
    def __init__(self, index_dir: Path, uploads_dir: Path, dim: int = 384):
        self.index_dir = Path(index_dir)
        self.uploads_dir = Path(uploads_dir)
        self.dim = dim 
        self._lock = RLock()
        self._index: VectorIndex | None = None 
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        
    def _ensure_index(self) -> VectorIndex:
        if self._index is None:
            idx = VectorIndex(dim=self.dim, storage_dir=str(self.index_dir))
            try:
                idx.load()
            except Exception as exc: # pragma: no cover - log only
                logger.warning("Failed to load index: %s", exc)
            self._index = idx
        return self._index
    
    
    def add_file(self, file_path: Path, doc_id: str) -> int:
        chunks = build_chunks(str(file_path), doc_id=doc_id)
        if not chunks:
            return 0
        
        texts: List[str] = []
        metas: List[Dict] = []
        for chunk in chunks:
            meta = dict(chunk.meta)
            meta["doc_id"] = chunk.doc_id
            meta["chunk_index"] = len(metas) + 1
            texts.append(chunk.text)
            metas.append(meta)
        
        with self._lock:
            index = self._ensure_index()
            index.add_texts(texts, metas)
            index.save()
        
        return len(texts)
    
    
    def search(self, query: str, k: int) -> List[Dict]:
        with self._lock:
            index = self._ensure_index()
            raw_hits = index.search(query, k=k)
            hits: List[Dict] = []
            for score, meta in raw_hits:
                record = dict(meta)
                record.setdefault("text", meta.get("text", ""))
                record["score"] = score 
                hits.append(record)
            return hits
        
    
    def is_empty(self) -> bool:
        with self._lock:
            index = self._ensure_index()
            return index.is_empty()
        
    
