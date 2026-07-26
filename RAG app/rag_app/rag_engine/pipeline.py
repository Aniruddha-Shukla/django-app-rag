import time
import os
from typing import List, Dict, Any
from .document_processor import DocumentProcessor
from .embedding_service import LocalEmbeddingService
from .vector_store import VectorStore
from .llm_service import GeminiLLMService

class RAGPipeline:
    """
    Unified Production RAG Pipeline Orchestrator.
    Manages document ingestion, vector indexing, similarity retrieval, and LLM response generation.
    """

    def __init__(self, db_dir: str):
        self.doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        self.embedding_service = LocalEmbeddingService()
        self.vector_store = VectorStore(
            db_dir=db_dir,
            collection_name="rag_knowledge_base",
            embedding_service=self.embedding_service
        )
        self.llm_service = GeminiLLMService()

    def ingest_text(self, text: str, source_name: str = "user_input") -> Dict[str, Any]:
        """Ingests raw text string into vector database."""
        start_time = time.time()
        chunks = self.doc_processor.chunk_text(text, source_name=source_name)
        count = self.vector_store.add_chunks(chunks)
        elapsed = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "source": source_name,
            "chunks_added": count,
            "latency_ms": elapsed,
            "total_chunks_in_db": self.vector_store.get_stats()["total_chunks"]
        }

    def ingest_file(self, file_path: str, source_name: str = None) -> Dict[str, Any]:
        """Ingests file (.txt, .md, .pdf) into vector database."""
        start_time = time.time()
        if not source_name:
            source_name = os.path.basename(file_path)

        text = self.doc_processor.extract_text_from_file(file_path)
        chunks = self.doc_processor.chunk_text(text, source_name=source_name)
        count = self.vector_store.add_chunks(chunks)
        elapsed = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "source": source_name,
            "chunks_added": count,
            "latency_ms": elapsed,
            "total_chunks_in_db": self.vector_store.get_stats()["total_chunks"]
        }

    def query(self, query_text: str, n_results: int = 4) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query workflow:
        1. Embed query & vector similarity search in ChromaDB.
        2. Format augmented prompt with top context.
        3. Call Gemini LLM generation.
        4. Track granular execution latency.
        """
        total_start = time.time()

        # Step 1: Retrieval
        retrieval_start = time.time()
        context_chunks = self.vector_store.query_similar(query_text, n_results=n_results)
        retrieval_latency = round((time.time() - retrieval_start) * 1000, 2)

        # Step 2: Generation
        gen_start = time.time()
        llm_response = self.llm_service.generate_rag_response(query_text, context_chunks)
        generation_latency = round((time.time() - gen_start) * 1000, 2)

        total_latency = round((time.time() - total_start) * 1000, 2)

        return {
            "query": query_text,
            "answer": llm_response["answer"],
            "sources": context_chunks,
            "model": llm_response["model_used"],
            "is_fallback": llm_response["is_fallback"],
            "metrics": {
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": generation_latency,
                "total_latency_ms": total_latency,
                "sources_retrieved": len(context_chunks)
            }
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Returns overall system health, vector database, and GPU engine stats."""
        stats = self.vector_store.get_stats()
        stats["gemini_api_configured"] = bool(os.getenv("GEMINI_API_KEY"))
        return stats

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clears all stored vector documents."""
        success = self.vector_store.clear()
        return {
            "status": "cleared" if success else "error",
            "total_chunks": 0
        }
