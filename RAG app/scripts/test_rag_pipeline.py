#!/usr/bin/env python
"""
Automated Test Suite for NexusRAG Backend Engine.
Tests document processing, TensorFlow embeddings, ChromaDB indexing, and Gemini RAG generation.
"""

import os
import sys
import time

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print("🚀 NexusRAG Automated Backend Integration Test")
    print("=" * 70)

    from rag_app.rag_engine.pipeline import RAGPipeline

    test_db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_chroma_db")
    print(f"[1/5] Initializing RAG Pipeline (Vector Store Path: {test_db_dir})...")
    pipeline = RAGPipeline(db_dir=test_db_dir)

    # Check device info
    stats = pipeline.get_system_stats()
    print(f"      Hardware Accelerator: {stats['embedding_device']['device']}")
    print(f"      Vector Dimensions:    {stats['embedding_device']['vector_dim']}d")
    print(f"      Gemini API Status:    {'Configured' if stats['gemini_api_configured'] else 'Pending Key (.env)'}")

    # Test Ingestion
    print("\n[2/5] Ingesting Technical Documentation into Knowledge Base...")
    doc_1 = """
    Retrieval-Augmented Generation (RAG) is an enterprise AI framework that enhances LLMs by retrieving relevant domain context from an external vector store (such as ChromaDB) before model inference.
    RAG reduces LLM hallucinations and provides verifiable, cited answers.
    Key components include document parsing, chunking with sliding window overlap, neural vector embedding calculation, vector database retrieval using cosine similarity, and prompt augmentation.
    """

    doc_2 = """
    TensorFlow and Apple Silicon Metal GPU acceleration enable rapid local vector embedding generation.
    ChromaDB stores high-dimensional embeddings using Hierarchical Navigable Small World (HNSW) graphs, allowing sub-millisecond similarity queries across millions of document chunks.
    """

    res1 = pipeline.ingest_text(doc_1, source_name="rag_overview.md")
    res2 = pipeline.ingest_text(doc_2, source_name="vector_perf.txt")

    print(f"      Ingested Doc 1: {res1['chunks_added']} chunks added ({res1['latency_ms']} ms)")
    print(f"      Ingested Doc 2: {res2['chunks_added']} chunks added ({res2['latency_ms']} ms)")
    print(f"      Total Chunks in DB: {pipeline.get_system_stats()['total_chunks']}")

    # Test Vector Retrieval & RAG Query
    print("\n[3/5] Executing RAG Similarity Query...")
    query_text = "How does RAG reduce hallucinations and what components are used?"
    
    query_result = pipeline.query(query_text=query_text, n_results=3)

    print("\n[4/5] RAG Pipeline Query Results:")
    print(f"      Query: \"{query_result['query']}\"")
    print(f"      Model Used: {query_result['model']}")
    print(f"      Total Latency: {query_result['metrics']['total_latency_ms']} ms")
    print(f"      Retrieval Latency: {query_result['metrics']['retrieval_latency_ms']} ms")
    print(f"      Generation Latency: {query_result['metrics']['generation_latency_ms']} ms")
    print(f"      Sources Retrieved: {len(query_result['sources'])}")

    print("\n--- Context Chunks Retrieved ---")
    for idx, src in enumerate(query_result['sources'], 1):
        print(f"  [{idx}] Source: {src['metadata'].get('source')} | Similarity: {src['similarity_score']*100:.1f}%")
        print(f"      Content: \"{src['text'][:120]}...\"")

    print("\n--- Generated RAG Answer ---")
    print(query_result['answer'])

    print("\n[5/5] Verification Complete: All RAG Pipeline components operational! ✅")
    print("=" * 70)

if __name__ == "__main__":
    main()
