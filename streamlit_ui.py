"""
streamlit_ui.py
DriveAgent v2 — Streamlit Demo UI
Run with: streamlit run streamlit_ui.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import requests

API_BASE = "http://localhost:8001"

st.set_page_config(
    page_title="DriveAgent v2",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 DriveAgent v2")
st.caption("Llama 3.2 · llama.cpp · INT4 Quantized · Map-Reduce Summarization · Google Drive")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_BASE)

    st.markdown("---")
    st.subheader("Pipeline")
    st.markdown("""
1. **Download** file from Drive
2. **Parse** — PDF/DOCX/PPTX/Image/Audio/Video
3. **Chunk** — 512-token sliding window
4. **Map-Reduce** — chunk summaries → doc summary
5. **Upload** — summary DOCX back to Drive
""")
    st.markdown("---")

    # Health check
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success("API online")
            col1, col2 = st.columns(2)
            col1.metric("CPU", f"{data.get('cpu_percent', 0)}%")
            col2.metric("RAM", f"{data.get('ram_used_gb', 0)}/{data.get('ram_total_gb', 0)} GB")
            st.info(f"Model: {data.get('slm_model', 'llama3.2')}")
            st.info(f"Engine: {data.get('inference_engine', 'ollama')}")
        else:
            st.error("API not responding")
    except Exception:
        st.error("API offline — run: python agent_loop.py --api")

# ── Main UI ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")

    file_id = st.text_input(
        "Google Drive File ID",
        value=st.session_state.get("selected_file_id", ""),
        placeholder="e.g. 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
        help="From Drive URL: drive.google.com/file/d/FILE_ID/view"
    )

    task = st.selectbox(
        "Task",
        options=["summarise", "ask", "extract", "edit"],
        format_func=lambda x: {
            "summarise": "📄 Summarise document",
            "ask":       "❓ Ask a question",
            "extract":   "🔍 Extract information",
            "edit":      "✏️  Edit document",
        }[x]
    )

    if task == "summarise":
        query = st.text_area("Instruction", value="Summarise this document in clear bullet points.", height=80)
        upload_summary = st.checkbox("Upload summary back to Drive as DOCX", value=False)
    elif task == "ask":
        query = st.text_area("Your question", placeholder="What are the key findings?", height=80)
        upload_summary = False
    elif task == "extract":
        query = st.text_area("What to extract", placeholder="Extract all names, dates, and figures.", height=80)
        upload_summary = False
    elif task == "edit":
        query = st.text_area("Edit instruction", placeholder="Fix all grammar errors.", height=80)
        upload_back = st.checkbox("Upload edited file back to Drive", value=True)
        upload_summary = False

    run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)

with col2:
    st.subheader("Response")

    if run_btn:
        if not file_id.strip():
            st.error("Please enter a Google Drive File ID.")
        elif not query.strip():
            st.error("Please enter an instruction.")
        else:
            with st.spinner("Processing... (map-reduce takes 2-4 minutes on CPU)"):
                try:
                    if task == "edit":
                        payload = {"file_id": file_id, "instruction": query,
                                   "upload_back": upload_back}
                        resp = requests.post(f"{api_url}/edit", json=payload, timeout=600)
                    else:
                        endpoint = {"summarise": "/summarise", "ask": "/ask",
                                    "extract": "/extract"}[task]
                        payload = {
                            "file_id": file_id,
                            "query": query,
                            "upload_summary": upload_summary if task == "summarise" else False,
                        }
                        resp = requests.post(f"{api_url}{endpoint}", json=payload, timeout=600)

                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✓ Processed: **{data.get('file_name', 'Unknown')}**")

                        # Metrics
                        meta_cols = st.columns(4)
                        mime = data.get("mime_type", "")
                        meta_cols[0].metric("File type", mime.split("/")[-1] if mime else "—")
                        meta_cols[1].metric("Chunks", data.get("chunks_indexed", "—"))

                        latency = data.get("latency", {})
                        if latency:
                            total_s = round(latency.get("total_ms", 0) / 1000, 1)
                            inf_s = round(latency.get("inference_ms", 0) / 1000, 1)
                            meta_cols[2].metric("Total time", f"{total_s}s")
                            meta_cols[3].metric("Inference", f"{inf_s}s")

                            with st.expander("📊 Latency breakdown"):
                                st.json(latency)
                        else:
                            meta_cols[2].metric("Status", "Done")
                            meta_cols[3].metric("Upload", data.get("status", "—"))

                        st.markdown("---")
                        st.markdown("**Result:**")

                        if task == "edit":
                            st.text_area("Edited content", value=data.get("edited_content", ""), height=300)
                        else:
                            response_text = data.get("response", "")
                            if response_text:
                                st.markdown(response_text)
                            else:
                                st.warning("Empty response. Try a different instruction.")

                        if data.get("output_file_name"):
                            st.success(f"📎 Summary uploaded to Drive: {data.get('output_file_name')}")

                    else:
                        try:
                            error_detail = resp.json().get("detail", "Unknown error")
                        except Exception:
                            error_detail = resp.text or "Unknown error"
                        st.error(f"API error {resp.status_code}: {error_detail}")

                except requests.exceptions.Timeout:
                    st.error("Timed out — model still processing. Try again or use a shorter document.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect. Run: python agent_loop.py --api")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Enter a File ID and task, then click Run Agent.")

# ── File Browser ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Browse Drive files"):
    if st.button("List files"):
        try:
            resp = requests.get(f"{api_url}/files", timeout=30)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                if files:
                    for f in files:
                        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
                        c1.write(f["name"])
                        c2.write(f.get("mimeType", "").split(".")[-1])
                        c3.write(f.get("size", "—"))
                        if c4.button("Use", key=f["id"]):
                            st.session_state["selected_file_id"] = f["id"]
                            st.rerun()
                else:
                    st.info("No files found.")
            else:
                st.error("Failed to list files.")
        except Exception as e:
            st.error(f"Error: {e}")
