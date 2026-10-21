"""
file_router.py
MIME-type detection, routes to correct parser, dedup cache.
Matches architecture spec: file_router.py
"""

import os
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = "./data/cache.db"

MIME_ROUTING = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "pptx",
    "application/pdf": "pdf",
    "image/jpeg": "image", "image/png": "image",
    "image/webp": "image", "image/gif": "image",
    "audio/mpeg": "audio", "audio/wav": "audio",
    "audio/mp4": "audio", "audio/x-m4a": "audio",
    "video/mp4": "video", "video/quicktime": "video",
    "video/x-msvideo": "video", "video/x-matroska": "video",
    "text/plain": "txt", "text/markdown": "txt",
    "text/html": "txt", "text/csv": "csv",
}

EXTENSION_ROUTING = {
    ".docx": "docx", ".doc": "docx",
    ".xlsx": "xlsx", ".xls": "xlsx", ".csv": "csv",
    ".pptx": "pptx", ".ppt": "pptx",
    ".pdf": "pdf",
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".webp": "image", ".gif": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
    ".mp4": "video", ".mov": "video", ".avi": "video", ".mkv": "video",
    ".txt": "txt", ".md": "txt", ".html": "txt",
}


def _init_cache():
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            file_id TEXT PRIMARY KEY,
            file_hash TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            chunk_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _get_file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_already_processed(file_id: str, file_hash: str) -> bool:
    """SQLite dedup cache — skip already-processed files."""
    _init_cache()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT file_hash FROM processed_files WHERE file_id=?", (file_id,)
    ).fetchone()
    conn.close()
    return row is not None and row[0] == file_hash


def mark_as_processed(file_id: str, file_hash: str, chunk_count: int):
    _init_cache()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO processed_files (file_id, file_hash, chunk_count) VALUES (?,?,?)",
        (file_id, file_hash, chunk_count)
    )
    conn.commit()
    conn.close()


def route_file(local_path: str, mime_type: str = "", file_id: str = "") -> str:
    """
    Detect file type and extract text content.
    Returns extracted plain text.
    """
    path = Path(local_path)
    ext = path.suffix.lower()

    extractor_type = MIME_ROUTING.get(mime_type) or EXTENSION_ROUTING.get(ext)

    if not extractor_type:
        print(f"Unknown file type: mime={mime_type}, ext={ext}. Trying plain text.")
        return _read_plain_text(local_path)

    print(f"Routing '{path.name}' → {extractor_type} parser")
    return _dispatch(extractor_type, local_path)


def _dispatch(extractor_type: str, local_path: str) -> str:
    if extractor_type == "docx":
        from parsers.docx_parser import extract_docx
        return extract_docx(local_path)
    elif extractor_type == "xlsx" or extractor_type == "csv":
        from parsers.csv_parser import extract_csv_xlsx
        return extract_csv_xlsx(local_path)
    elif extractor_type == "pptx":
        from parsers.pptx_parser import extract_pptx
        return extract_pptx(local_path)
    elif extractor_type == "pdf":
        from parsers.pdf_parser import extract_pdf
        return extract_pdf(local_path)
    elif extractor_type == "image":
        from parsers.image_parser import extract_image
        return extract_image(local_path)
    elif extractor_type in ("audio", "video"):
        from parsers.audio_parser import extract_audio_video
        return extract_audio_video(local_path)
    elif extractor_type == "txt":
        return _read_plain_text(local_path)
    else:
        return _read_plain_text(local_path)


def _read_plain_text(local_path: str) -> str:
    try:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read as plain text: {e}")
        return ""
