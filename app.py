"""
documentReader — Flask backend
================================
A professional server-side implementation of a Retrieval-Augmented
Generation (RAG) document Q&A app.

Pipeline (classic RAG):
  1. INGEST   -> extract raw text from an uploaded PDF / DOCX / TXT / MD file
  2. CHUNK    -> split the text into overlapping passages
  3. INDEX    -> vectorize passages with TF-IDF (scikit-learn), including
                 bigrams, so retrieval is topic- and phrase-aware
  4. RETRIEVE -> for each question, embed the query with the same
                 vectorizer and rank passages by cosine similarity
  5. AUGMENT  -> stuff the top-K passages into the LLM prompt as context
  6. GENERATE -> call an LLM (GroqCloud, OpenAI-compatible) constrained to
                 answer ONLY from the supplied passages

All document state lives server-side, keyed to a per-browser session id
(a signed cookie), so multiple users of the same deployment never see
each other's documents or API keys.
"""

import os
import re
import io
import uuid
import time
import logging
from datetime import datetime

import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader
import docx as docx_lib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB, mirrors the original client limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_reader")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

CHUNK_SIZE_WORDS = 130
CHUNK_OVERLAP_WORDS = 30
TOP_K = 3

# ----------------------------------------------------------------------
# In-memory, per-session document store
#   SESSIONS[session_id] = {
#       "api_key": str | None,
#       "next_id": int,
#       "documents": { doc_id: {...} }
#   }
# NOTE: this resets if the process restarts. For a production deployment
# with persistence across restarts, swap this for a database (see README).
# ----------------------------------------------------------------------

SESSIONS = {}


def get_session_store():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    sid = session["sid"]
    if sid not in SESSIONS:
        SESSIONS[sid] = {"api_key": DEFAULT_GROQ_API_KEY or None, "next_id": 1, "documents": {}}
    return SESSIONS[sid]


SAMPLE_DOC_NAME = "Nimbus Robotics — Q3 2026 Report (sample)"
SAMPLE_DOC = (
    "NIMBUS ROBOTICS INC. — INTERNAL Q3 2026 REPORT\n"
    "Prepared for: All-Hands Meeting, October 14, 2026\n"
    "Classification: Internal Use Only\n\n"
    "1. COMPANY OVERVIEW\n\n"
    "Nimbus Robotics Inc. was founded in March 2021 by Priya Chandrasekaran and Tomas Ericsson in Boulder, Colorado. "
    "The company designs and manufactures modular warehouse robots under the product line \"Hopper,\" aimed at "
    "small-to-mid-sized fulfillment centers that cannot afford full Amazon-scale automation. As of Q3 2026, Nimbus "
    "employs 214 people across three offices: Boulder (HQ, 140 employees), Austin (52 employees, manufacturing "
    "liaison), and a small Berlin sales office (22 employees).\n\n"
    "The company's mission statement, adopted at the 2023 board retreat, is: \"Make warehouse automation accessible "
    "to businesses with fewer than 500 employees.\"\n\n"
    "2. PRODUCT LINE\n\n"
    "The flagship product is the Hopper-3, a modular pick-and-carry robot released in June 2025. It succeeded the "
    "Hopper-2 (released January 2023) and the original Hopper-1 prototype (2021, never sold commercially). The "
    "Hopper-3 has a maximum payload of 68 kilograms, a battery life of 9.5 hours under typical warehouse load, and a "
    "recharge time of 47 minutes using the fast-dock system introduced in firmware version 4.2.\n\n"
    "In Q3 2026, Nimbus also launched a software-only product called \"Nimbus Fleet Console,\" a subscription-based "
    "dashboard that lets warehouse managers coordinate up to 40 Hopper units simultaneously. Fleet Console costs "
    "$1,200 per month per warehouse site, regardless of the number of robots deployed.\n\n"
    "3. FINANCIAL SUMMARY — Q3 2026\n\n"
    "Total revenue for Q3 2026 was $18.4 million, up 22% from Q2 2026's $15.1 million. Hardware sales (Hopper-3 "
    "units) accounted for $14.2 million of this total, while Fleet Console subscriptions and other software/services "
    "made up the remaining $4.2 million.\n\n"
    "Gross margin for the quarter was 41%, an improvement from 36% in Q2, attributed primarily to the renegotiated "
    "battery-cell supply contract signed with PowerCell Dynamics in July 2026.\n\n"
    "Nimbus raised a $60 million Series C round in April 2026, led by Meridian Growth Partners, with participation "
    "from existing investors Blue Anchor Ventures and Third Compass Capital. This brought total funding raised to "
    "date to $103 million. The company reported $41 million in cash reserves at the end of Q3 2026, giving an "
    "estimated runway of approximately 14 months at current burn rate if revenue growth were to stall.\n\n"
    "4. CUSTOMERS\n\n"
    "As of the end of Q3 2026, Nimbus has 37 active warehouse customers operating a combined fleet of 612 Hopper "
    "robots. The largest customer is Redwood Grocers Distribution, which operates 84 units across two facilities in "
    "Sacramento and Reno. The next-largest deployment is at Fenwick Auto Parts, with 61 units in a single Columbus, "
    "Ohio facility.\n\n"
    "Customer churn for the quarter was low: only one customer (a small e-commerce fulfillment startup called "
    "QuickCrate) discontinued service, citing a shift to a different automation vendor.\n\n"
    "5. HEADCOUNT AND HIRING\n\n"
    "Nimbus grew from 178 employees at the start of Q3 to 214 by the end of the quarter, a net addition of 36 "
    "people. Of these, 21 were engineering hires (mostly firmware and computer vision roles), 9 were in sales and "
    "customer success, and 6 were in manufacturing operations in Austin.\n\n"
    "The company plans to open a fourth office in late 2026 or early 2027, most likely in either Toronto or "
    "Chicago, to support expansion into the Great Lakes distribution corridor. No final decision on the new office "
    "location had been made as of the report date.\n\n"
    "6. PRODUCT ROADMAP\n\n"
    "The engineering team is currently developing the Hopper-4, targeted for a limited pilot release in Q2 2027. "
    "Planned improvements include a 15% increase in payload capacity (targeting 78 kilograms), swappable battery "
    "packs to reduce downtime during shift changes, and an upgraded LIDAR system sourced from a new supplier, "
    "Veyron Sensing.\n\n"
    "A secondary roadmap item is an API integration layer allowing Fleet Console to connect directly with major "
    "warehouse management systems (WMS), starting with SAP Extended Warehouse Management and Manhattan Associates' "
    "WMS platform. This integration is scheduled for beta testing in January 2027.\n\n"
    "7. RISKS AND CHALLENGES\n\n"
    "The report flags three primary risks for Q4 2026 and beyond:\n\n"
    "First, battery-cell supply remains a bottleneck despite the improved PowerCell Dynamics contract; a "
    "single-supplier dependency could affect production if PowerCell experiences disruptions.\n\n"
    "Second, two competitors — Atlas Warehouse Systems and a stealth-mode startup rumored to be backed by a major "
    "logistics company — are reportedly targeting the same mid-market segment Nimbus serves.\n\n"
    "Third, the report notes that Nimbus has not yet achieved profitability and continues to operate at a net "
    "loss, though the loss narrowed to $2.1 million in Q3 2026 compared to $3.8 million in Q2 2026.\n\n"
    "8. CLOSING NOTES FROM LEADERSHIP\n\n"
    "CEO Priya Chandrasekaran closed the report with a note thanking the team for hitting the Q3 revenue target and "
    "reiterating the company's goal of reaching operating profitability by Q4 2027. CTO Tomas Ericsson added a "
    "brief technical note that the Hopper-4's computer vision stack will move from the current third-party model to "
    "an in-house model trained on Nimbus's own warehouse imagery dataset, which now exceeds 2.3 million labeled "
    "images."
)

# ----------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected and could not be read.")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    if not text.strip():
        raise ValueError(
            "Could not extract text from this PDF. It may be scanned/image-only, which requires OCR."
        )
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        document = docx_lib.Document(io.BytesIO(file_bytes))
    except Exception:
        raise ValueError("Could not parse this .docx file. It may be corrupted or in an unsupported format.")
    paragraphs = [p.text for p in document.paragraphs]
    text = "\n\n".join(paragraphs)
    if not text.strip():
        raise ValueError("This document appears to be empty, or its text could not be extracted.")
    return text


def extract_text_from_plain(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("Could not read this file's text encoding.")


def file_type_label(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "PDF"
    if lower.endswith(".docx"):
        return "DOCX"
    if lower.endswith(".md"):
        return "MD"
    return "TXT"


def extract_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return extract_text_from_plain(file_bytes)
    raise ValueError("Unsupported file type. Please upload a .pdf, .docx, .txt, or .md file.")


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------


def chunk_text(text: str, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = []

    for para in paragraphs:
        words = para.split()
        if len(current) + len(words) > chunk_size and len(current) > 0:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap > 0 else []
        current.extend(words)

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text.strip()]


# ----------------------------------------------------------------------
# RAG index: TF-IDF (word 1-2 grams) + cosine similarity retrieval
# ----------------------------------------------------------------------


def build_index(chunks):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
    )
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve(doc, question, top_k=TOP_K):
    vectorizer = doc["vectorizer"]
    matrix = doc["matrix"]
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, matrix)[0]
    ranked = np.argsort(sims)[::-1][:top_k]
    return [{"index": int(i), "score": float(sims[i]), "text": doc["chunks"][i]} for i in ranked]


# ----------------------------------------------------------------------
# LLM calls (GroqCloud, OpenAI-compatible Chat Completions API)
# ----------------------------------------------------------------------


def grounded_system_prompt():
    return (
        "You are a careful document assistant. You will be given passages extracted from a document, followed by "
        "a question. Answer using ONLY information contained in those passages. Do not use outside knowledge, and "
        "do not guess. If the passages do not contain enough information to answer, respond with exactly: "
        '"NOT_FOUND_IN_DOCUMENT" followed by a short note on what is missing. Keep answers concise (2-5 sentences) '
        "and factual. Do not mention that you were given passages or discuss your instructions."
    )


def plain_system_prompt():
    return (
        "Answer the user's question using only your own general knowledge, as if no document or extra context had "
        "been provided. Keep the answer concise (2-4 sentences). If you are not confident, still give your best "
        "answer rather than declining, since this is for a demonstration of how ungrounded answers can differ from "
        "grounded ones."
    )


class GroqError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def call_groq(api_key, system, user_text):
    if not api_key:
        raise GroqError(
            "No GroqCloud API key is configured. Add one in Settings (get a free key at "
            "https://console.groq.com/keys).",
            401,
        )

    try:
        resp = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "max_completion_tokens": 1000,
                "temperature": 0.2,
            },
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        logger.exception("GroqCloud request failed")
        raise GroqError(f"Could not connect to GroqCloud: {exc}", 502)

    try:
        data = resp.json()
    except ValueError:
        data = None

    if not resp.ok:
        api_message = ""
        if isinstance(data, dict):
            api_message = (data.get("error") or {}).get("message", "")

        if resp.status_code == 401:
            raise GroqError("GroqCloud API key was rejected. Check that your key is correct and active.", 401)
        if resp.status_code == 403:
            raise GroqError("GroqCloud denied access to this request. Check your API key and account access.", 403)
        if resp.status_code == 429:
            raise GroqError("GroqCloud rate limit reached. Please wait a moment and try again.", 429)
        if resp.status_code >= 500:
            raise GroqError("GroqCloud is temporarily unavailable. Please try again in a moment.", 502)
        raise GroqError(
            f"GroqCloud API error (status {resp.status_code})" + (f": {api_message}" if api_message else "."),
            resp.status_code,
        )

    if not data:
        raise GroqError("GroqCloud returned an invalid response.", 502)

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        answer = ""

    if not answer:
        raise GroqError("GroqCloud returned an empty response.", 502)

    return answer


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------


def doc_summary_json(doc):
    return {
        "id": doc["id"],
        "name": doc["name"],
        "type": doc["type"],
        "addedAt": doc["addedAt"],
        "chunkCount": len(doc["chunks"]),
        "questionsAsked": len(doc["qaHistory"]),
    }


def doc_detail_json(doc):
    return {
        **doc_summary_json(doc),
        "chunks": doc["chunks"],
        "qaHistory": doc["qaHistory"],
        "hallucinationFlags": doc["hallucinationFlags"],
    }


def require_doc(store, doc_id):
    doc = store["documents"].get(doc_id)
    if not doc:
        return None
    return doc


# ----------------------------------------------------------------------
# Routes — pages
# ----------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------
# Routes — settings (API key)
# ----------------------------------------------------------------------


@app.route("/api/settings/api-key", methods=["GET"])
def get_api_key_status():
    store = get_session_store()
    return jsonify({"hasKey": bool(store["api_key"])})


@app.route("/api/settings/api-key", methods=["POST"])
def set_api_key():
    store = get_session_store()
    payload = request.get_json(silent=True) or {}
    key = (payload.get("apiKey") or "").strip()
    if not key:
        store["api_key"] = None
        return jsonify({"hasKey": False})
    store["api_key"] = key
    return jsonify({"hasKey": True})


# ----------------------------------------------------------------------
# Routes — documents
# ----------------------------------------------------------------------


@app.route("/api/documents", methods=["GET"])
def list_documents():
    store = get_session_store()
    docs = sorted(store["documents"].values(), key=lambda d: d["addedAt"])
    return jsonify({"documents": [doc_summary_json(d) for d in docs]})


@app.route("/api/documents/<int:doc_id>", methods=["GET"])
def get_document(doc_id):
    store = get_session_store()
    doc = require_doc(store, doc_id)
    if not doc:
        return jsonify({"error": "Document not found."}), 404
    return jsonify(doc_detail_json(doc))


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    store = get_session_store()
    if doc_id in store["documents"]:
        del store["documents"][doc_id]
        return jsonify({"ok": True})
    return jsonify({"error": "Document not found."}), 404


def _ingest_document(store, name, text):
    if not text or not text.strip():
        raise ValueError("That document appears to be empty, or its text could not be extracted.")

    chunks = chunk_text(text)
    vectorizer, matrix = build_index(chunks)

    doc_id = store["next_id"]
    store["next_id"] += 1

    doc = {
        "id": doc_id,
        "name": name,
        "type": file_type_label(name),
        "addedAt": int(time.time() * 1000),
        "rawText": text,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "qaHistory": [],
        "hallucinationFlags": [],
    }
    store["documents"][doc_id] = doc
    return doc


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    store = get_session_store()

    if "file" not in request.files:
        return jsonify({"error": "No file was provided."}), 400

    file = request.files["file"]
    filename = file.filename or "document"
    lower = filename.lower()
    if not (lower.endswith(".pdf") or lower.endswith(".docx") or lower.endswith(".txt") or lower.endswith(".md")):
        return jsonify({"error": "Unsupported file type. Please upload a .pdf, .docx, .txt, or .md file."}), 400

    file_bytes = file.read()
    if len(file_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "That file is larger than 6 MB. Try a shorter document."}), 400

    try:
        text = extract_text(filename, file_bytes)
        doc = _ingest_document(store, filename, text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        logger.exception("Failed to ingest uploaded document")
        return jsonify({"error": "Something went wrong reading that file."}), 500

    return jsonify(doc_detail_json(doc)), 201


@app.route("/api/documents/sample", methods=["POST"])
def add_sample_document():
    store = get_session_store()
    doc = _ingest_document(store, SAMPLE_DOC_NAME, SAMPLE_DOC)
    return jsonify(doc_detail_json(doc)), 201


# ----------------------------------------------------------------------
# Routes — RAG Q&A
# ----------------------------------------------------------------------


@app.route("/api/documents/<int:doc_id>/ask", methods=["POST"])
def ask_question(doc_id):
    store = get_session_store()
    doc = require_doc(store, doc_id)
    if not doc:
        return jsonify({"error": "Document not found."}), 404

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    compare = bool(payload.get("compare"))

    if not question:
        return jsonify({"error": "A question is required."}), 400

    retrieved = retrieve(doc, question, TOP_K)
    context_text = "\n\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(retrieved))
    user_prompt = f"Passages:\n\n{context_text}\n\nQuestion: {question}"

    try:
        grounded_raw = call_groq(store["api_key"], grounded_system_prompt(), user_prompt)
        plain_text = call_groq(store["api_key"], plain_system_prompt(), question) if compare else None
    except GroqError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    not_found = grounded_raw.startswith("NOT_FOUND_IN_DOCUMENT")
    if not_found:
        answer_text = re.sub(r"^[:\-\s]+", "", grounded_raw.replace("NOT_FOUND_IN_DOCUMENT", "", 1))
        answer_text = answer_text or "This isn't covered in the document."
    else:
        answer_text = grounded_raw

    qa_index = len(doc["qaHistory"])
    doc["qaHistory"].append(
        {
            "question": question,
            "groundedAnswer": answer_text,
            "notFound": not_found,
            "plainAnswer": plain_text,
            "retrievalIndices": [r["index"] for r in retrieved],
            "retrievalScores": [r["score"] for r in retrieved],
        }
    )

    return jsonify(
        {
            "qaIndex": qa_index,
            "question": question,
            "groundedAnswer": answer_text,
            "notFound": not_found,
            "plainAnswer": plain_text,
            "retrieved": retrieved,
            "questionsAsked": len(doc["qaHistory"]),
        }
    )


@app.route("/api/documents/<int:doc_id>/flag", methods=["POST"])
def flag_answer(doc_id):
    store = get_session_store()
    doc = require_doc(store, doc_id)
    if not doc:
        return jsonify({"error": "Document not found."}), 404

    payload = request.get_json(silent=True) or {}
    qa_index = payload.get("qaIndex")
    if qa_index is None or qa_index < 0 or qa_index >= len(doc["qaHistory"]):
        return jsonify({"error": "Invalid question index."}), 400

    if qa_index in doc["hallucinationFlags"]:
        doc["hallucinationFlags"].remove(qa_index)
        flagged = False
    else:
        doc["hallucinationFlags"].append(qa_index)
        flagged = True

    return jsonify({"flagged": flagged})


@app.route("/api/documents/<int:doc_id>/summary", methods=["POST"])
def generate_summary(doc_id):
    store = get_session_store()
    doc = require_doc(store, doc_id)
    if not doc:
        return jsonify({"error": "Document not found."}), 404
    if not doc["qaHistory"]:
        return jsonify({"error": "Ask at least one question before generating a summary."}), 400

    lines = []
    for i, qa in enumerate(doc["qaHistory"]):
        lines.append(f"{i + 1}. Q: {qa['question']}")
        grounded = f"[not found in document] {qa['groundedAnswer']}" if qa["notFound"] else qa["groundedAnswer"]
        lines.append(f"   Grounded answer: {grounded}")
        top_score = qa["retrievalScores"][0] if qa["retrievalScores"] else None
        lines.append(f"   Top retrieval score: {top_score:.3f}" if top_score is not None else "   Top retrieval score: n/a")
        if qa["plainAnswer"]:
            flagged = i in doc["hallucinationFlags"]
            suffix = "  [FLAGGED BY USER AS POSSIBLE HALLUCINATION]" if flagged else ""
            lines.append(f"   Ungrounded answer: {qa['plainAnswer']}{suffix}")
    transcript = "\n\n".join(lines)

    system = (
        "You are helping write a short project summary for a RAG (Retrieval-Augmented Generation) mini-project. "
        "You will be given a transcript of questions asked against a document, the grounded (document-based) "
        "answers with retrieval confidence scores, and in some cases an ungrounded answer for comparison, possibly "
        "flagged by the user as a suspected hallucination. Write a concise summary (250-400 words) covering: (1) "
        "how grounding in the document changed answer quality vs a plain/ungrounded prompt, citing 1-2 specific "
        "examples from the transcript, (2) any flagged or likely hallucinations and why they happened, (3) a "
        "one-line takeaway. Write in plain prose, no headers, no markdown formatting, professional but not stiff."
    )
    user_text = f"Document name: {doc['name']}\nNumber of chunks indexed: {len(doc['chunks'])}\n\nTranscript:\n\n{transcript}"

    try:
        summary = call_groq(store["api_key"], system, user_text)
    except GroqError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"summary": summary})


# ----------------------------------------------------------------------
# Error handlers
# ----------------------------------------------------------------------


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "That file is larger than 6 MB. Try a shorter document."}), 413


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found."}), 404
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
