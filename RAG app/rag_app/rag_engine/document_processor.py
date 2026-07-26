import os
import re
from typing import List, Dict, Any
import pypdf

class DocumentProcessor:
    """
    Modular Document Processor for loading, parsing, and chunking text documents.
    Supports .txt, .md, .pdf files as well as raw text input.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract plain text from a file based on its extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.txt', '.md', '.markdown', '.json', '.csv', '.py']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif ext == '.pdf':
            text_content = []
            reader = pypdf.PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text_content.append(f"[Page {i+1}]\n" + extracted)
            return "\n\n".join(text_content)

        else:
            # Fallback text loader
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def chunk_text(self, text: str, source_name: str = "raw_input") -> List[Dict[str, Any]]:
        """
        Splits text into overlapping chunks with rich metadata.
        """
        # Clean text
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)
        chunk_idx = 0

        while start < text_length:
            end = start + self.chunk_size

            # Try to break at natural paragraph or sentence boundaries if possible
            if end < text_length:
                # Look for natural breaks (paragraph, newline, period, space)
                break_pos = -1
                for boundary in ['\n\n', '\n', '. ', ' ']:
                    found = text.rfind(boundary, start + self.chunk_overlap, end)
                    if found != -1 and found > start:
                        break_pos = found + (len(boundary) if boundary in ['. ', ' '] else 0)
                        break

                if break_pos != -1:
                    end = break_pos

            chunk_str = text[start:end].strip()

            if chunk_str:
                chunks.append({
                    "id": f"{source_name}_chunk_{chunk_idx}",
                    "text": chunk_str,
                    "metadata": {
                        "source": source_name,
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_str),
                        "start_char": start,
                        "end_char": end
                    }
                })
                chunk_idx += 1

            # Move window with overlap
            if end >= text_length:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks
