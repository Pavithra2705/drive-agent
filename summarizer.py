"""
summarizer.py
Map-reduce summarization orchestration.
Matches architecture spec: summarizer.py — map-reduce summarization,
chunk-level → doc-level → multi-doc, custom prompt library.
"""

import slm_engine


def summarize_document(chunks: list[dict], query: str = "") -> str:
    """
    Map-reduce summarization:
    1. Summarise each chunk individually (map)
    2. Combine chunk summaries into final document summary (reduce)
    """
    if not chunks:
        return "No content to summarise."

    print(f"Map-reduce: summarising {len(chunks)} chunks...")

    # --- MAP phase: summarise each chunk ---
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarising chunk {i+1}/{len(chunks)}...")
        summary = slm_engine.summarize(chunk["text"])
        if summary:
            source_tag = f"[Chunk {i+1}]"
            chunk_summaries.append(f"{source_tag} {summary}")

    if not chunk_summaries:
        return "Could not generate summaries."

    # --- REDUCE phase: combine into final summary ---
    if len(chunk_summaries) == 1:
        return chunk_summaries[0].replace("[Chunk 1] ", "")

    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = f"""You are a document summariser.
Below are summaries of different sections of a document.
Combine them into ONE final coherent summary.

SECTION SUMMARIES:
{combined}

Write a final unified summary. Cover all key points. Do not repeat.
Use bullet points. Be concise.

FINAL SUMMARY:"""

    print("Reducing chunk summaries to final document summary...")
    final = slm_engine.generate(reduce_prompt, max_new_tokens=512)
    return final


def summarize_for_query(chunks: list[dict], query: str) -> str:
    """
    Query-focused summarization — retrieve relevant chunks
    and answer the user's specific question.
    """
    if not chunks:
        return "No relevant content found."

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return slm_engine.qa(context, query)
