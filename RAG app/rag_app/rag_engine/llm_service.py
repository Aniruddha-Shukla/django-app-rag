import os
from typing import List, Dict, Any

class GeminiLLMService:
    """
    LLM Generation Service utilizing Google Gemini API.
    Handles prompt construction, RAG grounding instructions, and response generation.
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client if API key is available."""
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            print("[GeminiLLMService] Warning: GEMINI_API_KEY is not set. Operating in fallback synthesis mode.")
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            print("[GeminiLLMService] Google GenAI SDK client initialized successfully.")
        except Exception as e:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.client = "legacy"
                print("[GeminiLLMService] Legacy Google GenerativeAI SDK initialized.")
            except Exception as legacy_err:
                print(f"[GeminiLLMService] Client initialization error: {e} / {legacy_err}")

    def generate_rag_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a grounded RAG response combining user query and retrieved context chunks.
        """
        # Format context text block with source numbering
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, 1):
            source_name = chunk.get("metadata", {}).get("source", f"Document {idx}")
            chunk_text = chunk.get("text", "")
            score = chunk.get("similarity_score", 0.0)
            context_blocks.append(f"--- [Source {idx}: {source_name} | Similarity: {score * 100:.1f}%] ---\n{chunk_text}")

        formatted_context = "\n\n".join(context_blocks) if context_blocks else "No relevant background context found in local vector storage."

        system_instruction = (
            "You are an expert Retrieval-Augmented Generation (RAG) Assistant.\n"
            "Your goal is to provide accurate, comprehensive, and clear answers based directly on the retrieved context below.\n"
            "Guidelines:\n"
            "1. Base your answer strictly on the provided background context.\n"
            "2. Cite your sources using bracket notation e.g., [Source 1], [Source 2] where appropriate.\n"
            "3. If the context does not contain enough information to answer the query, clearly state what information is missing based on what you retrieved.\n"
            "4. Keep your tone professional, structured, and easy to read."
        )

        user_prompt = (
            f"BACKGROUND CONTEXT:\n{formatted_context}\n\n"
            f"USER QUERY: {query}\n\n"
            f"Please answer the user query based on the context above."
        )

        # Call Gemini API if client is available
        if self.client and self.client != "legacy":
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.3,
                    }
                )
                return {
                    "answer": response.text,
                    "model_used": self.model_name,
                    "is_fallback": False
                }
            except Exception as e:
                print(f"[GeminiLLMService] Gemini API call error: {e}")
                # Fallback to synthesis mode

        elif self.client == "legacy":
            try:
                import google.generativeai as legacy_genai
                model = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(user_prompt)
                return {
                    "answer": response.text,
                    "model_used": f"{self.model_name} (Legacy SDK)",
                    "is_fallback": False
                }
            except Exception as e:
                print(f"[GeminiLLMService] Legacy Gemini API error: {e}")

        # Intelligent RAG Synthesis Fallback if API key is not configured
        synthesized_answer = self._synthesize_fallback_answer(query, context_chunks)
        return {
            "answer": synthesized_answer,
            "model_used": "Local RAG Synthesis Engine (Add GEMINI_API_KEY in .env for live Gemini 2.5 Flash)",
            "is_fallback": True
        }

    def _synthesize_fallback_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Synthesizes structured grounded output from context chunks when API key is pending."""
        if not context_chunks:
            return (
                "**No Context Available in Knowledge Base**\n\n"
                "I couldn't find any relevant documents or text chunks in the local ChromaDB vector store matching your query.\n\n"
                "💡 **Action Required**: Please add text or upload a document using the Knowledge Base sidebar to populate the RAG database!"
            )

        top_sources = []
        for idx, chunk in enumerate(context_chunks[:3], 1):
            src = chunk.get("metadata", {}).get("source", "Document")
            sim = chunk.get("similarity_score", 0.0)
            top_sources.append(f"- **[Source {idx}: {src}]** (Relevance: {sim*100:.1f}%)\n  > \"{chunk.get('text', '')[:200]}...\"")

        sources_str = "\n\n".join(top_sources)

        return (
            f"### RAG Local Context Summary\n\n"
            f"Based on your query: *\"{query}\"*, the local ChromaDB vector store retrieved the top matching context chunks:\n\n"
            f"{sources_str}\n\n"
            f"--- \n"
            f"💡 **Notice**: To connect live **Google Gemini 2.5 Flash** generation, add your `GEMINI_API_KEY` in the `.env` file!"
        )
