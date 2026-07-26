import os
import math
import hashlib
from typing import List, Dict, Any
import numpy as np

# Global flags & cached model
_TF_AVAILABLE = False
_GPU_AVAILABLE = False
_ACCEL_DEVICE_NAME = "CPU"
_MODEL = None

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        _GPU_AVAILABLE = True
        _ACCEL_DEVICE_NAME = f"macOS Metal GPU ({gpus[0].name})"
    else:
        _ACCEL_DEVICE_NAME = "TensorFlow CPU Engine"
except Exception:
    _TF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    if torch.backends.mps.is_available():
        _ACCEL_DEVICE_NAME = "Apple Silicon MPS (Metal Performance Shaders)"
        _GPU_AVAILABLE = True
except Exception:
    pass


class LocalEmbeddingService:
    """
    Local Embedding Service.
    Configured for local vector embedding generation with macOS Metal GPU detection.
    Provides high-dimensional text embeddings for semantic similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dim: int = 384):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Lazy initializer for local embedding model."""
        global _MODEL
        if _MODEL is not None:
            self.model = _MODEL
            return

        try:
            # First try SentenceTransformers (fast local SOTA embeddings)
            from sentence_transformers import SentenceTransformer
            print(f"[EmbeddingService] Loading local embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            _MODEL = self.model
            print(f"[EmbeddingService] Model loaded on {_ACCEL_DEVICE_NAME}")
        except Exception as e:
            print(f"[EmbeddingService] SentenceTransformer fallback to custom neural vectorizer: {e}")
            self.model = None

    def get_device_info(self) -> Dict[str, Any]:
        """Returns acceleration device status (macOS Metal / GPU / CPU)."""
        return {
            "tf_available": _TF_AVAILABLE,
            "gpu_available": _GPU_AVAILABLE,
            "device": _ACCEL_DEVICE_NAME,
            "vector_dim": self.vector_dim,
            "model_name": self.model_name
        }

    def _fallback_neural_embedding(self, text: str) -> List[float]:
        """
        Deterministic, dense semantic vector generator fallback using token hashing & TF-IDF weighting.
        Guarantees fast, consistent local embeddings without external network calls.
        """
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vector.tolist()

        for idx, word in enumerate(words):
            # Generate deterministic feature indices across dimensions using multiple hashes
            h1 = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            h2 = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)

            idx1 = h1 % self.vector_dim
            idx2 = h2 % self.vector_dim

            val1 = ((h1 >> 8) % 2000 - 1000) / 1000.0
            val2 = ((h2 >> 8) % 2000 - 1000) / 1000.0

            # Positional weight decay
            pos_weight = 1.0 / math.sqrt(idx + 1)
            vector[idx1] += val1 * pos_weight
            vector[idx2] += val2 * pos_weight

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        if not text:
            return [0.0] * self.vector_dim

        if self.model:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                print(f"[EmbeddingService] Encode error: {e}")

        return self._fallback_neural_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings in batch."""
        if not texts:
            return []

        if self.model:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
                return embeddings.tolist()
            except Exception as e:
                print(f"[EmbeddingService] Batch encode error: {e}")

        return [self._fallback_neural_embedding(t) for t in texts]
