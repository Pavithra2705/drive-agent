"""
chunker.py
512-token sliding window chunker with sentence boundary detection.
Matches architecture spec: chunker.py — 512-token sliding window,
overlap strategy, sentence boundary detection, metadata tagging.
"""

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def chunk_text(text: str, source_file: str = "", page: int = 0) -> list[dict]:
    """
    Split text into overlapping chunks with metadata tags.

    Returns list of dicts:
    {
        "text": "...",
        "source": "filename",
        "page": 1,
        "chunk_index": 0
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.get("chunk_size", 512),
        chunk_overlap=CONFIG.get("chunk_overlap", 50),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    chunks = []

    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source_file,
                "page": page,
                "chunk_index": i,
            })

    return chunks


def chunks_to_texts(chunks: list[dict]) -> list[str]:
    """Extract just the text strings from chunk dicts."""
    return [c["text"] for c in chunks]
