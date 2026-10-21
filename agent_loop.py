"""
agent_loop.py
Main orchestrator — scheduler, retry logic, logging, CLI + API interface.
Matches architecture spec: agent_loop.py — main entry point,
watch-mode daemon, CLI + API interface.

Usage:
  python agent_loop.py --watch                    # Watch Drive folder continuously
  python agent_loop.py --file <drive_file_id>     # Process single file
  python agent_loop.py --api                      # Start FastAPI server
"""

import argparse
import time
import yaml
import os
import sys
import shutil
import tempfile
import sqlite3

# Load config
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Setup logging
import logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, CONFIG.get("log_level", "INFO")),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG.get("log_path", "./logs/agent.log")),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def process_file(drive, file_id: str, upload_summary: bool = True) -> dict:
    """
    Full pipeline for a single file:
    1. Download
    2. Parse & extract text
    3. Chunk with metadata tags
    4. Map-reduce summarize
    5. Upload summary back to Drive
    """
    import file_router
    import chunker
    import summarizer
    import output_writer

    # Get metadata
    meta = drive.get_file_metadata(file_id)
    mime_type = meta["mimeType"]
    file_name = meta["name"]

    logger.info(f"Processing: '{file_name}' ({mime_type})")

    # Check dedup cache
    tmpdir = tempfile.mkdtemp()
    try:
        # Download
        local_path = os.path.join(tmpdir, file_name)
        local_path = drive.download_file(file_id, local_path)

        # Check if already processed (SQLite dedup)
        from file_router import _get_file_hash, is_already_processed, mark_as_processed
        file_hash = _get_file_hash(local_path)

        if is_already_processed(file_id, file_hash):
            logger.info(f"Skipping '{file_name}' — already processed (dedup cache)")
            return {"status": "skipped", "reason": "already_processed", "file_name": file_name}

        # Extract text
        t = time.time()
        extracted_text = file_router.route_file(local_path, mime_type, file_id)
        if not extracted_text or not extracted_text.strip():
            logger.warning(f"Could not extract content from '{file_name}'")
            return {"status": "error", "reason": "extraction_failed", "file_name": file_name}
        logger.info(f"Extracted {len(extracted_text)} chars in {time.time()-t:.1f}s")

        # Chunk with metadata
        t = time.time()
        chunks = chunker.chunk_text(extracted_text, source_file=file_name)
        logger.info(f"Chunked into {len(chunks)} pieces in {time.time()-t:.1f}s")

        # Map-reduce summarize
        t = time.time()
        summary = summarizer.summarize_document(chunks)
        logger.info(f"Summary generated in {time.time()-t:.1f}s")

        # Write and upload back to Drive
        result = {"status": "success", "file_name": file_name, "summary": summary}

        if upload_summary:
            metadata = {
                "chunks": len(chunks),
                "chars_extracted": len(extracted_text),
                "file_type": mime_type,
            }
            upload_result = output_writer.write_and_upload(
                drive, file_id, file_name, summary, metadata
            )
            result.update(upload_result)

        # Mark as processed in SQLite cache
        mark_as_processed(file_id, file_hash, len(chunks))
        logger.info(f"Done: '{file_name}' → summary uploaded")

        return result

    except Exception as e:
        logger.error(f"Failed to process '{file_name}': {e}")
        return {"status": "error", "reason": str(e), "file_name": file_name}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def watch_mode(drive):
    """
    Watch Drive folder — polls every N seconds and processes new files.
    Matches spec: Polls Drive API every N minutes or uses push notifications.
    """
    interval = CONFIG.get("watch_interval", 300)
    folder_id = CONFIG.get("drive_folder_id", "root")

    logger.info(f"Watch mode started — polling every {interval}s")
    logger.info(f"Watching folder: {folder_id}")

    processed_ids = set()

    while True:
        try:
            files = drive.list_files(folder_id=folder_id)
            new_files = [f for f in files if f["id"] not in processed_ids]

            if new_files:
                logger.info(f"Found {len(new_files)} new file(s)")
                for f in new_files:
                    result = process_file(drive, f["id"])
                    processed_ids.add(f["id"])
                    logger.info(f"Result: {result['status']} — {f['name']}")
            else:
                logger.info("No new files found")

        except KeyboardInterrupt:
            logger.info("Watch mode stopped")
            break
        except Exception as e:
            logger.error(f"Watch loop error: {e}")

        logger.info(f"Sleeping {interval}s...")
        time.sleep(interval)


def start_api():
    """Start FastAPI server."""
    import uvicorn
    from api import app
    uvicorn.run(
        app,
        host=CONFIG.get("api_host", "0.0.0.0"),
        port=CONFIG.get("api_port", 8001),
        log_level=CONFIG.get("log_level", "info").lower(),
    )


def main():
    parser = argparse.ArgumentParser(description="DriveAgent — SLM-powered Google Drive AI Agent")
    parser.add_argument("--watch", action="store_true", help="Watch Drive folder continuously")
    parser.add_argument("--file", type=str, help="Process a single Drive file ID")
    parser.add_argument("--api", action="store_true", help="Start FastAPI REST server")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload summary back to Drive")
    args = parser.parse_args()

    from drive_connector import DriveConnector
    drive = DriveConnector()

    if args.watch:
        watch_mode(drive)
    elif args.file:
        result = process_file(drive, args.file, upload_summary=not args.no_upload)
        print("\nResult:")
        print(f"  Status  : {result.get('status')}")
        print(f"  File    : {result.get('file_name')}")
        if result.get("summary"):
            print(f"\nSummary:\n{result['summary']}")
    elif args.api:
        start_api()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
