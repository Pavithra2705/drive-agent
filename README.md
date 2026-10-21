# DriveAgent v2
**SLM-Powered Google Drive AI Agent**
Llama 3.2 · llama.cpp · INT4 Quantized · Map-Reduce · Multi-Modal

---

## Architecture (matches specification)

```
INPUT LAYER          — Google Drive Files (PDF, PPTX, DOCX, Images, Videos, Audio, CSV)
        ↓
PARSING LAYER        — parsers/pdf_parser, docx_parser, pptx_parser, image_parser, audio_parser
        ↓
CHUNKING             — chunker.py (512-token sliding window, metadata tagging)
        ↓
SLM INFERENCE        — slm_engine.py (Llama 3.2 via Ollama/llama.cpp, INT4 quantized)
        ↓
MAP-REDUCE           — summarizer.py (chunk → section → document summary)
        ↓
OUTPUT LAYER         — output_writer.py (DOCX/JSON/TXT → upload back to Drive)
```

---

## Project Structure

```
driveagent/
├── agent_loop.py        # Main orchestrator — watch mode, CLI, scheduler
├── drive_connector.py   # Google Drive API v3 — OAuth2, list/download/upload
├── file_router.py       # MIME detection, parser dispatch, SQLite dedup cache
├── chunker.py           # 512-token sliding window chunker with metadata
├── summarizer.py        # Map-reduce summarization (chunk → doc level)
├── slm_engine.py        # Llama 3.2 via Ollama (llama.cpp backend)
├── output_writer.py     # Write DOCX/JSON summary, upload to Drive
├── api.py               # FastAPI REST interface
├── setup_oauth.py       # One-time Google OAuth setup
├── config.yaml          # All configuration settings
├── requirements.txt     # Dependencies
│
├── parsers/
│   ├── pdf_parser.py    # PyMuPDF — PDF text + tables
│   ├── docx_parser.py   # python-docx — Word documents
│   ├── pptx_parser.py   # python-pptx — PowerPoint slides
│   ├── image_parser.py  # BLIP-2 caption + TrOCR OCR
│   ├── audio_parser.py  # Whisper STT — audio + video
│   └── csv_parser.py    # Pandas/openpyxl — CSV/Excel
│
├── models/              # GGUF model files (auto-managed by Ollama)
├── data/
│   └── cache.db         # SQLite dedup cache
└── logs/
    └── agent.log        # Processing logs
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Llama 3.2 model
```bash
ollama pull llama3.2
```

### 3. Google OAuth setup (one-time)
```bash
python setup_oauth.py
```

### 4. Run the agent

**Watch Drive folder continuously:**
```bash
python agent_loop.py --watch
```

**Process a single file:**
```bash
python agent_loop.py --file YOUR_DRIVE_FILE_ID
```

**Start REST API:**
```bash
python agent_loop.py --api
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | System status + CPU/RAM metrics |
| GET | /files | List Drive files |
| POST | /summarise | Summarise any file |
| POST | /ask | Ask a question about a file |
| POST | /extract | Extract structured info |
| POST | /edit | Edit and re-upload file |

---

## Key Features

- **Llama 3.2 3B** — INT4 quantized via Ollama/llama.cpp, runs on CPU
- **Map-Reduce** — chunk summaries combined into final document summary
- **SQLite dedup** — already-processed files are skipped automatically
- **Watch mode** — polls Drive every N seconds, processes new files
- **Multi-modal** — PDF, DOCX, PPTX, images, audio, video, CSV
- **Write-back** — summary uploaded as DOCX to Google Drive
- **Zero cloud API** — 100% local, private, offline-capable
