import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from .embedding_service import LocalEmbeddingService

class VectorStore:
    """
    ChromaDB Local Vector Storage Manager.
    Handles persistent database storage, document indexing, and nearest-neighbor vector retrieval.
    """

    def __init__(self, db_dir: str, collection_name: str = "rag_knowledge_base", embedding_service: Optional[LocalEmbeddingService] = None):
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_service = embedding_service or LocalEmbeddingService()

        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Adds text chunks to ChromaDB.
        Generates local embeddings before insertion.
        """
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate embeddings locally via GPU/accelerated service
        embeddings = self.embedding_service.get_embeddings(texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(chunks)

    def query_similar(self, query_text: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for the top K most relevant text chunks based on cosine similarity.
        """
        if not query_text or self.collection.count() == 0:
            return []

        # Generate query vector
        query_embedding = self.embedding_service.get_embedding(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for i in range(len(docs)):
                # Cosine distance to similarity score conversion
                dist = distances[i]
                similarity_score = round(max(0.0, 1.0 - dist), 4)

                retrieved.append({
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": round(dist, 4),
                    "similarity_score": similarity_score
                })

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection stats and count."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "db_path": self.db_dir,
            "embedding_device": self.embedding_service.get_device_info()
        }

    def clear(self) -> bool:
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"[VectorStore] Clear error: {e}")
            return False
