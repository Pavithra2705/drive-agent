"""
output_writer.py
Result formatter and Drive uploader.
Matches architecture spec: output_writer.py — TXT/DOCX/JSON output,
summary + key entities, write back to Drive.
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
import yaml

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def write_and_upload(
    drive,
    original_file_id: str,
    original_file_name: str,
    summary: str,
    metadata: dict = None,
) -> dict:
    """
    Write summary to file and upload back to Google Drive.
    Returns dict with output file info.
    """
    output_format = CONFIG.get("output_format", "docx")
    suffix = CONFIG.get("output_suffix", "_summary")
    base_name = Path(original_file_name).stem
    output_name = f"{base_name}{suffix}"

    tmpdir = tempfile.mkdtemp()

    try:
        if output_format == "docx":
            output_path = os.path.join(tmpdir, f"{output_name}.docx")
            _write_docx(output_path, original_file_name, summary, metadata)
        elif output_format == "json":
            output_path = os.path.join(tmpdir, f"{output_name}.json")
            _write_json(output_path, original_file_name, summary, metadata)
        else:
            output_path = os.path.join(tmpdir, f"{output_name}.txt")
            _write_txt(output_path, original_file_name, summary, metadata)

        # Upload to Drive
        new_file_id = drive.upload_file(output_path)
        print(f"Summary uploaded to Drive: {output_name}")

        return {
            "output_file_id": new_file_id,
            "output_file_name": Path(output_path).name,
            "output_format": output_format,
            "status": "uploaded",
        }

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_docx(path: str, source_name: str, summary: str, metadata: dict = None):
    """Write summary as a formatted DOCX file."""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        # Title
        title = doc.add_heading("Document Summary", level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Metadata
        doc.add_paragraph(f"Source: {source_name}")
        doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        doc.add_paragraph("")

        # Summary heading
        doc.add_heading("Summary", level=2)

        # Summary content
        for line in summary.split("\n"):
            line = line.strip()
            if line:
                if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
                    p = doc.add_paragraph(line[2:], style="List Bullet")
                else:
                    doc.add_paragraph(line)

        # Metadata section
        if metadata:
            doc.add_paragraph("")
            doc.add_heading("Processing Details", level=2)
            for key, value in metadata.items():
                doc.add_paragraph(f"{key}: {value}")

        doc.save(path)
        print(f"DOCX written: {path}")

    except Exception as e:
        print(f"DOCX write failed: {e}")
        _write_txt(path.replace(".docx", ".txt"), source_name, summary, metadata)


def _write_txt(path: str, source_name: str, summary: str, metadata: dict = None):
    """Write summary as plain text."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"DOCUMENT SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"Source: {source_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"SUMMARY\n{'-'*30}\n")
        f.write(summary)
        if metadata:
            f.write(f"\n\nPROCESSING DETAILS\n{'-'*30}\n")
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")
    print(f"TXT written: {path}")


def _write_json(path: str, source_name: str, summary: str, metadata: dict = None):
    """Write summary as structured JSON."""
    output = {
        "source_file": source_name,
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "metadata": metadata or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"JSON written: {path}")
