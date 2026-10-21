"""
api.py
FastAPI REST interface for DriveAgent.
Matches architecture spec: FastAPI — optional REST interface.
"""

import os
import time
import shutil
import tempfile
from typing import Optional
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="DriveAgent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_inference_lock = Lock()
_drive = None


def get_drive():
    global _drive
    if _drive is None:
        from drive_connector import DriveConnector
        _drive = DriveConnector()
    return _drive


class FileRequest(BaseModel):
    file_id: str
    query: str = "Summarise this document in clear bullet points."
    task_type: str = "summarise"
    upload_summary: bool = False


class EditRequest(BaseModel):
    file_id: str
    instruction: str
    upload_back: bool = True


@app.get("/health")
def health_check():
    try:
        import psutil
        import slm_engine
        return {
            "status": "ok",
            "slm_loaded": slm_engine.is_loaded(),
            "slm_model": "llama3.2",
            "inference_engine": "ollama/llama.cpp",
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        }
    except Exception as e:
        return {"status": "ok", "error": str(e)}


@app.get("/files")
def list_files(folder_id: Optional[str] = None, page_size: int = 20):
    try:
        files = get_drive().list_files(folder_id=folder_id, page_size=page_size)
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarise")
def summarise(req: FileRequest):
    return _run_pipeline(req.file_id, req.query, "summarise", req.upload_summary)


@app.post("/ask")
def ask(req: FileRequest):
    return _run_pipeline(req.file_id, req.query, "qa", False)


@app.post("/extract")
def extract(req: FileRequest):
    return _run_pipeline(req.file_id, req.query, "extract", False)


@app.post("/edit")
def edit(req: EditRequest):
    import file_router
    import slm_engine
    import output_writer

    drive = get_drive()
    try:
        meta = drive.get_file_metadata(req.file_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")

    mime_type = meta["mimeType"]
    file_name = meta["name"]
    tmpdir = tempfile.mkdtemp()

    try:
        local_path = os.path.join(tmpdir, file_name)
        local_path = drive.download_file(req.file_id, local_path)
        extracted_text = file_router.route_file(local_path, mime_type)

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract content.")

        prompt = f"""You are a document editor.
Edit the following document according to the instruction.
Preserve all original content not targeted by the edit.

ORIGINAL DOCUMENT:
{extracted_text[:3000]}

EDIT INSTRUCTION: {req.instruction}

EDITED DOCUMENT:"""

        with _inference_lock:
            edited_content = slm_engine.generate(prompt, max_new_tokens=512)

        result = {
            "file_id": req.file_id,
            "file_name": file_name,
            "edited_content": edited_content,
        }

        if req.upload_back:
            from agent.edit_agent import EditAgent
            edit_result = EditAgent(drive).apply_and_upload(
                file_id=req.file_id,
                local_path=local_path,
                mime_type=mime_type,
                edited_text=edited_content,
                instruction=req.instruction,
            )
            result.update(edit_result)

        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_pipeline(file_id: str, query: str, task_type: str, upload_summary: bool) -> dict:
    import file_router
    import chunker
    import summarizer
    import slm_engine
    import output_writer

    timings = {}
    overall_start = time.time()
    drive = get_drive()

    # Metadata
    t = time.time()
    try:
        meta = drive.get_file_metadata(file_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")
    timings["metadata_ms"] = round((time.time() - t) * 1000)

    mime_type = meta["mimeType"]
    file_name = meta["name"]
    print(f"Processing: '{file_name}' ({mime_type})")

    tmpdir = tempfile.mkdtemp()
    try:
        # Download
        t = time.time()
        local_path = os.path.join(tmpdir, file_name)
        local_path = drive.download_file(file_id, local_path)
        timings["download_ms"] = round((time.time() - t) * 1000)

        # Extract
        t = time.time()
        extracted_text = file_router.route_file(local_path, mime_type, file_id)
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract content from this file.")
        timings["extraction_ms"] = round((time.time() - t) * 1000)

        # Chunk
        t = time.time()
        chunks = chunker.chunk_text(extracted_text, source_file=file_name)
        timings["chunking_ms"] = round((time.time() - t) * 1000)

        # Summarise or QA
        t = time.time()
        with _inference_lock:
            if task_type == "summarise":
                response = summarizer.summarize_document(chunks, query)
            else:
                response = summarizer.summarize_for_query(chunks[:5], query)
        timings["inference_ms"] = round((time.time() - t) * 1000)
        timings["total_ms"] = round((time.time() - overall_start) * 1000)

        print(f"Latency: {timings}")

        result = {
            "file_id": file_id,
            "file_name": file_name,
            "mime_type": mime_type,
            "chunks_indexed": len(chunks),
            "response": response,
            "latency": timings,
        }

        # Upload summary back to Drive if requested
        if upload_summary and task_type == "summarise":
            upload_result = output_writer.write_and_upload(
                drive, file_id, file_name, response,
                {"chunks": len(chunks), "total_ms": timings["total_ms"]}
            )
            result.update(upload_result)

        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "status": "failed"}
    )
