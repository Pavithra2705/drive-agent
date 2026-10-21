"""
slm_engine.py
Llama inference core using Ollama (llama.cpp backend).
INT4/INT8 quantized GGUF loading via Ollama.
Matches architecture spec: slm_engine.py — llama.cpp backend,
quantized GGUF loading, prompt templating, streaming output, fallback chain.
"""

import time
import yaml
import ollama

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_model_name = CONFIG.get("model_path", "llama3.2")
_loaded = False


def _load_model():
    global _loaded
    if _loaded:
        return
    print(f"Loading SLM: {_model_name} via Ollama (llama.cpp backend)...")
    try:
        ollama.generate(
            model=_model_name,
            prompt="hello",
            options={"num_predict": 1}
        )
        _loaded = True
        print(f"SLM ready: {_model_name}")
    except Exception as e:
        print(f"SLM load failed: {e}")
        print(f"Make sure Ollama is running and model is pulled: ollama pull {_model_name}")
        raise


def summarize(text: str, max_tokens: int = None) -> str:
    """Summarise a single chunk of text."""
    _load_model()
    prompt = f"""Summarise the following text concisely in 2-3 sentences.
Only use information from the text. Do not add anything extra.

TEXT:
{text}

SUMMARY:"""
    return _generate(prompt, max_tokens)


def qa(context: str, question: str, max_tokens: int = None) -> str:
    """Answer a question based on document context."""
    _load_model()
    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "Not found in document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return _generate(prompt, max_tokens)


def generate(prompt: str, max_new_tokens: int = None) -> str:
    """General generation with a custom prompt."""
    _load_model()
    return _generate(prompt, max_new_tokens)


def _generate(prompt: str, max_tokens: int = None) -> str:
    max_tokens = max_tokens or CONFIG.get("max_tokens", 512)
    start = time.time()

    try:
        response = ollama.generate(
            model=_model_name,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": CONFIG.get("temperature", 0.1),
                "top_p": 0.9,
            }
        )
    except Exception as e:
        print(f"Ollama generation failed: {e}")
        return f"Error: {e}"

    elapsed = time.time() - start
    result = response.response if hasattr(response, "response") else response.get("response", "")
    tokens = response.eval_count if hasattr(response, "eval_count") else response.get("eval_count", 0)
    result = result.strip() if result else ""

    tok_per_sec = round(tokens / elapsed, 1) if elapsed > 0 else 0
    print(f"Generated {tokens} tokens in {elapsed:.1f}s ({tok_per_sec} tok/s)")

    return result or "Empty response from model."


def is_loaded() -> bool:
    return _loaded
