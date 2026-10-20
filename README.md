# Unce GPT — ChatGPT-inspired workspace

Run with Python 3.10+:

```bash
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Choose OpenRouter (API key required) or local Ollama (install Ollama separately and pull a model). For Ollama, enter the exact installed model name.

Features: streamed responses, SQLite chat history, server-side uploaded files, safe in-memory ZIP inspection (no execution or extraction to disk), PDF/DOCX/text parsing, file workspace, ChatGPT-inspired responsive UI. Each file is limited to 20 MB. Archive text extraction is capped to prevent decompression bombs; binary formats are stored but not interpreted. Files persist under data/ and are not automatically deleted. Local-only, single-user development server: do not expose to the public internet without authentication, access control and stronger isolation.

Limitations: no semantic vector search, automatic code execution, integrated Git diff, or full IDE. Larger archives are truncated. The OpenRouter key is held in the browser tab and transmitted to the local backend per request, not persisted to disk. Markdown renderer uses CDNs and needs internet. Ollama model installation is not automatic.

## Large repository retrieval (Final edition)
Uploaded ZIPs and text documents are indexed in persistent SQLite FTS5, split into overlapping ~2,800-character passages. Chat requests retrieve the top 14 keyword-relevant passages (with filenames and chunk numbers), rather than injecting all files into the model context. `/api/search?q=authentication` lets you inspect results; `/api/reindex` rebuilds indexes for files uploaded in earlier editions. Retrieval works offline without a vector API. FTS5 is **lexical retrieval**, not embedding-based semantic similarity; exact concept matches using different vocabulary may be missed. Source uploads remain limited to 20 MB and text extraction to 3 million characters per file. Run locally only; authentication and hardened deployment are not included.

## Hybrid semantic retrieval and symbol index

For local embeddings install `pip install -r requirements-semantic.txt` (downloads a sentence-transformers model on first use; runs on CPU). Start the server normally, upload projects, then POST `/api/embeddings/reindex` to create persistent local vectors. New uploads are keyword-indexed immediately; run the embeddings reindex again after adding new files. `/api/search/hybrid?q=...` combines vector similarity and FTS5 with reciprocal-rank fusion. `/api/symbols?q=...` finds indexed Python classes/functions and common JavaScript/TypeScript/Go/Rust/Java declarations. The chat endpoint uses hybrid retrieval automatically, falling back to keyword search when embeddings are not installed or indexed. Vectors are stored in SQLite and searched with a simple exact cosine scan; this is functional but large repositories may need an ANN index for speed. Uploaded code is never executed. Use locally, not as a public multi-user service without authentication and upload isolation.
