from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch

class EmbeddingManager:
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        # trust_remote_code may be required for some Gemma models
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)

    def encode(self, texts, batch_size: int = 8):
        # returns numpy array shape (n, dim)
        embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # normalize for cosine search
        faiss.normalize_L2(embs)
        return embs
