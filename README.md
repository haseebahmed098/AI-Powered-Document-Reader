# documentReader — Flask + RAG

A Flask backend for the documentReader app. All document parsing, chunking,
retrieval, and LLM calls now happen **server-side**, and the browser only
renders UI and talks to a JSON API — no API keys or document text ever sit
in browser storage.

## What changed from the original single-file HTML

| | Before (browser-only) | Now (Flask) |
|---|---|---|
| PDF/DOCX parsing | pdf.js / mammoth.js in the browser | `pypdf` / `python-docx` on the server |
| Chunking + TF-IDF index | hand-rolled JS | `scikit-learn` `TfidfVectorizer` + `cosine_similarity` |
| Document storage | JS variable, lost on refresh | server-side, per-session store |
| GroqCloud API key | `localStorage` in the browser (visible to any script on the page) | held only in the signed server session, never sent to the client |
| LLM calls | browser → GroqCloud directly | browser → Flask → GroqCloud |

The retrieval step (TF-IDF + cosine similarity over document chunks) plus
the generation step (an LLM prompted to answer *only* from the retrieved
passages, with an explicit "not found in document" fallback) together form
the RAG pipeline. See `app.py`, functions `build_index`, `retrieve`, and
`call_groq` / `ask_question`.

## Project layout

```
document_reader_flask/
├── app.py                 # Flask app: routes, RAG pipeline, Groq calls
├── requirements.txt
├── .env.example            # copy to .env and fill in
├── templates/
│   └── index.html          # page shell (Jinja)
└── static/
    ├── css/style.css       # design system (colors, layout, components)
    └── js/app.js           # UI logic — calls /api/* only
```

## Running it locally

```bash
cd document_reader_flask
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env: set FLASK_SECRET_KEY to a random string.
# GROQ_API_KEY is optional — leave blank and set the key from the
# "API Key" button in the UI instead, if you'd rather not put it in .env.

python3 app.py
# → http://127.0.0.1:5000
```

Get a free GroqCloud API key at https://console.groq.com/keys — the app
uses their OpenAI-compatible Chat Completions endpoint with the
`llama-3.3-70b-versatile` model.

## Running in production

The dev server (`python3 app.py`) is not meant for production traffic. Use
a WSGI server, e.g.:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

`gunicorn` is already pinned in `requirements.txt`. Put a reverse proxy
(nginx, Caddy, etc.) in front of it for TLS.

**Important:** the current document store (`SESSIONS` in `app.py`) is a
plain in-memory Python dict. That's fine for a demo or single-process
deployment, but it means:
- documents are lost if the process restarts
- it will **not** work correctly with more than one worker process/replica,
  since each process has its own memory

For anything beyond a demo, swap `SESSIONS` for a real store — e.g. Redis
for the session/API-key data, and a database (Postgres, SQLite) for
document text, chunks, and Q&A history. The functions `get_session_store`,
`_ingest_document`, `require_doc`, and the route handlers are the only
places that would need to change; the RAG logic (`build_index`, `retrieve`)
is independent of storage.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/documents` | list documents in this session |
| POST | `/api/documents/upload` | upload + parse + chunk + index a file (multipart `file` field) |
| POST | `/api/documents/sample` | load the built-in sample report |
| GET | `/api/documents/<id>` | full document detail (chunks, Q&A history) |
| DELETE | `/api/documents/<id>` | remove a document |
| POST | `/api/documents/<id>/ask` | ask a question — runs retrieval + generation |
| POST | `/api/documents/<id>/flag` | toggle the "possible hallucination" flag on a Q&A turn |
| POST | `/api/documents/<id>/summary` | generate a project summary from the Q&A transcript |
| GET / POST | `/api/settings/api-key` | check / set the GroqCloud key for this session |

All error responses are JSON: `{"error": "human-readable message"}` with an
appropriate HTTP status code (400/401/403/404/422/429/502).

## Notes on the retrieval quality

TF-IDF with word 1–2 grams is a solid, dependency-light baseline for RAG
over a handful of short-to-medium documents, and it's what the original
app used (via hand-rolled JS). If you outgrow it — larger corpora, need for
semantic (not just keyword) matches — swap `build_index`/`retrieve` for a
dense embedding model (e.g. `sentence-transformers`) and a vector index
(e.g. FAISS); the rest of the app (chunking, prompting, UI) doesn't need
to change.
