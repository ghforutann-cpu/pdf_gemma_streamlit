import faiss
import numpy as np
from pathlib import Path

class FaissStore:
    def __init__(self, index_path: Path = Path("index.faiss"), meta_path: Path = Path("metadata.npy")):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index = None
        self.metadata = []

    def build_index(self, embeddings: np.ndarray, metadata: list):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.metadata = metadata
        faiss.write_index(index, str(self.index_path))
        np.save(self.meta_path, np.array(self.metadata, dtype=object))

    def load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = list(np.load(self.meta_path, allow_pickle=True))
            return True
        return False

    def search(self, query_embeddings: np.ndarray, top_k: int = 5):
        if self.index is None:
            self.load()
        D, I = self.index.search(query_embeddings, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append({"score": float(score), "meta": meta})
        return results

    def get_metadata_by_page(self, filename: str, page_number: int):
        # metadata entries include filename and page_number
        if not self.metadata:
            self.load()
        for m in self.metadata:
            if m.get("filename") == filename and m.get("page_number") == page_number:
                return m
        return None
