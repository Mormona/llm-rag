# app/main.py — RAG mini with intent, transform+RRF, shaping, hallucination filter, policy gate

import os, io, re, json, math, time, string, sqlite3, hashlib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
import numpy as np
import requests

# --------- Config ---------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_BASE = "https://api.mistral.ai/v1"
EMBED_MODEL_CANDIDATES = ["mistral-embed-v0.2", "mistral-embed"]
CHAT_MODEL_DEFAULT = "mistral-small"  # valid, fast
ENABLE_EVIDENCE_CHECK = os.getenv("ENABLE_EVIDENCE_CHECK", "0") == "1"

DB_PATH = "rag.db"
DOC_DIR = "storage"
os.makedirs(DOC_DIR, exist_ok=True)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 8
FINAL_K = 4
SEMANTIC_WEIGHT = 0.6
SIM_THRESHOLD = 0.22  # raise to 0.25 for stricter "insufficient evidence"
MAX_CHARS_PER_BLOCK = 800  # trim each evidence block for speed

# --------- FastAPI ---------
app = FastAPI(title="RAG Mini")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
def health():
    return {"status": "ok", "has_key": bool(MISTRAL_API_KEY)}

@app.get("/", response_class=HTMLResponse)
def root():
    return '<meta http-equiv="refresh" content="0; url=/ui" />'

@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception:
        return HTMLResponse("<h1>UI not found</h1><p>Create static/index.html</p>", status_code=200)

# --------- DB helpers ---------
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db():
    with db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY,
            doc_hash TEXT UNIQUE,
            filename TEXT,
            title TEXT,
            n_chunks INTEGER,
            created_at REAL
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            idx INTEGER,
            text TEXT,
            embedding TEXT,   -- JSON list[float]
            tf_json TEXT,     -- JSON {term:count}
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS vocab(
            term TEXT PRIMARY KEY,
            df INTEGER
        )""")
init_db()

# --------- Tiny tokenizer/TF-IDF ---------
STOP = set("a an and are as at be by for from has he in is it its of on or that the to was were will with you your".split())

def tokenize(txt: str) -> List[str]:
    txt = txt.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in txt.split() if t and t not in STOP]

def tf_counts(txt: str) -> Dict[str, int]:
    d = {}
    for t in tokenize(txt):
        d[t] = d.get(t, 0) + 1
    return d

def update_df(con, terms: List[str]):
    for t in set(terms):
        cur = con.execute("SELECT df FROM vocab WHERE term=?", (t,)).fetchone()
        if cur:
            con.execute("UPDATE vocab SET df=? WHERE term=?", (cur[0] + 1, t))
        else:
            con.execute("INSERT INTO vocab(term, df) VALUES (?, ?)", (t, 1))

def all_chunk_count(con) -> int:
    return int(con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

def idf(con, term: str, total_chunks: int) -> float:
    row = con.execute("SELECT df FROM vocab WHERE term=?", (term,)).fetchone()
    df = row[0] if row else 0
    if df == 0:
        return 0.0
    return math.log((1 + total_chunks) / (1 + df)) + 1.0

# --------- Mistral API helpers ---------
def _mistral_headers():
    if not MISTRAL_API_KEY:
        raise HTTPException(500, "Missing MISTRAL_API_KEY (set it in Space → Settings → Secrets).")
    return {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

def chat(messages, model: str = CHAT_MODEL_DEFAULT, temperature=0.0, max_tokens=700) -> str:
    r = requests.post(
        f"{MISTRAL_BASE}/chat/completions",
        headers=_mistral_headers(),
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=120
    )
    if r.status_code >= 400:
        raise HTTPException(500, f"Mistral chat error: {r.text}")
    return r.json()["choices"][0]["message"]["content"].strip()

def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """Embed texts in batches to respect Mistral API limits."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        last_err = None
        for model in EMBED_MODEL_CANDIDATES:
            try:
                r = requests.post(
                    f"{MISTRAL_BASE}/embeddings",
                    headers=_mistral_headers(),
                    json={"model": model, "input": batch},
                    timeout=60
                )
                if r.status_code < 400:
                    data = r.json()["data"]
                    all_embs.extend([d["embedding"] for d in data])
                    break
                last_err = r.text
            except Exception as e:
                last_err = str(e)
        else:
            raise HTTPException(500, f"Mistral embeddings failed: {last_err}")
        # polite pacing for free tier
        time.sleep(0.03)
    return all_embs

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))

# --------- PDF → chunks ---------
def read_pdf_bytes(b: bytes) -> str:
    reader = PdfReader(io.BytesIO(b))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    # normalize whitespace
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    # clamp overlap to be strictly less than size
    if overlap >= size:
        overlap = max(0, size // 5)  # fallback to 20% if misconfigured
    step = max(1, size - overlap)

    out = []
    i, L = 0, len(text)
    while i < L:
        chunk = text[i:i + size]
        # try not to cut mid-sentence for large chunks
        if len(chunk) > int(size * 0.6):
            last_period = chunk.rfind(". ")
            if last_period != -1 and last_period > int(size * 0.4):
                chunk = chunk[:last_period + 1]
        out.append(chunk.strip())
        i += step
    return [c for c in out if c]

# --------- Persist ---------
def save_document(filename: str, raw: bytes, chunks: List[str]):
    """Insert a new document; skip if same hash already exists."""
    h = hashlib.sha256(raw).hexdigest()
    with db() as con:
        row = con.execute("SELECT id FROM documents WHERE doc_hash=?", (h,)).fetchone()
        if row:
            return {"doc_id": row[0], "skipped": True, "n_chunks": 0}

        con.execute(
            "INSERT INTO documents(doc_hash, filename, title, n_chunks, created_at) VALUES (?, ?, ?, ?, ?)",
            (h, filename, os.path.splitext(filename)[0], len(chunks), time.time())
        )
        doc_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

        if chunks:
            embs = embed_texts(chunks)
            for idx, (chunk, emb) in enumerate(zip(chunks, embs)):
                tf = tf_counts(chunk)
                con.execute(
                    "INSERT INTO chunks(doc_id, idx, text, embedding, tf_json) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, idx, chunk, json.dumps(emb), json.dumps(tf))
                )
                update_df(con, list(tf.keys()))
    return {"doc_id": doc_id, "skipped": False, "n_chunks": len(chunks)}

# --------- Hybrid search ---------
def hybrid_search(query: str, top_k=TOP_K) -> List[Dict[str, Any]]:
    q_emb = np.array(embed_texts([query])[0], dtype=np.float32)
    q_tf = tf_counts(query)

    with db() as con:
        total = all_chunk_count(con)
        rows = con.execute("""
            SELECT chunks.id, chunks.doc_id, chunks.idx, chunks.text, chunks.embedding, documents.title
            FROM chunks JOIN documents ON chunks.doc_id = documents.id
        """).fetchall()

    # precompute idf for query terms once
    with db() as con2:
        total2 = all_chunk_count(con2)
        idf_cache = {t: idf(con2, t, total2) for t in q_tf}

    res = []
    for cid, doc_id, idx, text, emb_json, title in rows:
        # quick prefilter: skip chunks with no overlapping tokens to reduce compute
        if not (set(tf_counts(text).keys()) & set(q_tf.keys())):
            # Still compute semantic to avoid missing paraphrases
            pass
        emb = np.array(json.loads(emb_json), dtype=np.float32)
        sem = cosine(q_emb, emb)

        tfc = tf_counts(text)
        kw = 0.0
        # TF-IDF dot product (query vs chunk)
        for t, qcnt in q_tf.items():
            idf_t = idf_cache.get(t, 0.0)
            kw += (qcnt * idf_t) * (tfc.get(t, 0) * idf_t)

        res.append({"chunk_id": cid, "doc_id": doc_id, "idx": idx,
                    "title": title, "text": text, "semantic": sem, "keyword": kw})

    if not res:
        return []

    # normalize keyword and compute hybrid
    mx = max(r["keyword"] for r in res) or 1.0
    for r in res:
        r["keyword_n"] = r["keyword"] / mx
        r["hybrid"] = SEMANTIC_WEIGHT * r["semantic"] + (1.0 - SEMANTIC_WEIGHT) * r["keyword_n"]

    res.sort(key=lambda r: r["hybrid"], reverse=True)
    return res[:top_k]

# --------- Query helpers (intent, transform, shaping, policy, hallucination) ---------
INTENT_SMALLTALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b", re.I)

def is_smalltalk(q: str) -> bool:
    return bool(INTENT_SMALLTALK.match(q or ""))

def transform_query_basic(q: str) -> str:
    """Very light transform to improve keyword match signal."""
    q = (q or "").lower()
    q = q.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in q.split() if t and t not in STOP]
    return " ".join(toks[:8]) or q

def rrf_merge(hits_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    rank = {}
    merged = {}
    for hits in hits_list:
        for i, h in enumerate(hits):
            key = (h["title"], h["idx"])
            rank[key] = rank.get(key, 0) + 1.0 / (k + i + 1)
            if key not in merged:
                merged[key] = h
    out = list(merged.values())
    out.sort(key=lambda r: rank[(r["title"], r["idx"])], reverse=True)
    return out

SENSITIVE_HINTS = {
    "pii": ["social security", "ssn", "credit card", "password", "passport number"],
    "medical": ["diagnose", "treatment plan", "prescribe", "dose", "dosing"],
    "legal": ["create contract", "legal advice", "sue", "lawsuit", "liable"]
}

def policy_gate(q: str) -> Optional[str]:
    ql = (q or "").lower()
    if any(term in ql for term in SENSITIVE_HINTS["pii"]):
        return "I can’t help with requests that involve personal sensitive identifiers (PII)."
    if any(term in ql for term in SENSITIVE_HINTS["medical"]):
        return "I’m not a medical professional; I can’t provide medical advice. Consider consulting a qualified clinician."
    if any(term in ql for term in SENSITIVE_HINTS["legal"]):
        return "I’m not a lawyer; I can’t provide legal advice. Consider consulting a qualified attorney."
    return None

def choose_style(q: str) -> str:
    ql = (q or "").lower()
    if any(w in ql for w in ["list", "steps", "bullets", "bullet", "enumerate"]):
        return "list"
    if any(w in ql for w in ["table", "tabular", "columns"]):
        return "table"
    return "default"

def evidence_check(answer: str, uniq_chunks: List[Dict[str, Any]]) -> str:
    """
    Light post-hoc check:
    - Each sentence must include at least one [C#]
    - At least one cited chunk must share a token with the sentence
    If nothing passes, fall back to the original answer (avoid over-deleting).
    """
    if not (answer or "").strip():
        return answer
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    kept = []
    for s in sentences:
        tags = re.findall(r"\[C(\d+)\]", s)
        if not tags:
            continue
        sent_tokens = set(tokenize(s))
        ok = False
        for t in tags:
            i = int(t) - 1
            if 0 <= i < len(uniq_chunks):
                chunk_tokens = set(tokenize(uniq_chunks[i].get("text") or ""))
                if sent_tokens & chunk_tokens:
                    ok = True
                    break
        if ok:
            kept.append(s)
    return " ".join(kept) if kept else answer


def evidence_filter(answer: str) -> str:
    """Keep only sentences that cite evidence [C#]; else return 'insufficient evidence'."""
    parts = re.split(r'(?<=[.!?])\s+', (answer or "").strip())
    kept = [p for p in parts if re.search(r"\[C\d+\]", p)]
    return " ".join(kept) if kept else "insufficient evidence"

# --------- API models ---------
class QueryIn(BaseModel):
    query: str

# --------- Routes ---------
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided")

    out = []
    for f in files:
        data = await f.read()
        # save raw for reference
        with open(os.path.join(DOC_DIR, f.filename), "wb") as w:
            w.write(data)
        text = read_pdf_bytes(data)
        if not text.strip():
            out.append({"file": f.filename, "n_chunks": 0, "note": "No extractable text (maybe scanned PDF)."})
            continue
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        res = save_document(f.filename, data, chunks)
        out.append({"file": f.filename, "n_chunks": res["n_chunks"], "skipped": res["skipped"]})
    return {"status": "ok", "ingested": out}

@app.post("/query")
def query(q: QueryIn):
    user_q = (q.query or "").strip()
    if not user_q:
        return {"answer": "Please type a question.", "citations": []}

    # ---- policy + smalltalk gates ----
    guard = policy_gate(user_q)
    if guard:
        return {"answer": guard, "citations": []}
    if is_smalltalk(user_q):
        return {"answer": "Hi! Upload PDFs, then ask about them.", "citations": []}

    # ---- retrieval: original + transformed, then RRF fuse ----
    hits_a = hybrid_search(user_q, top_k=TOP_K)
    tq = transform_query_basic(user_q)
    hits_b = hybrid_search(tq, top_k=TOP_K) if tq != user_q else []
    hits = rrf_merge([hits_a, hits_b] if hits_b else [hits_a])

    # accept only if we have a reasonably similar candidate
    if not hits or max(h["semantic"] for h in hits) < SIM_THRESHOLD:
        return {"answer": "insufficient evidence", "citations": []}

    # ---- adaptive selection ----
    TINY_THRESHOLD = 5
    selected = hits if len(hits) <= TINY_THRESHOLD else hits[:FINAL_K]

    # de-duplicate by (title, idx)
    uniq, seen = [], set()
    for c in selected:
        key = (c["title"], c["idx"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    if not uniq:
        return {"answer": "insufficient evidence", "citations": []}

    # ---- build evidence context (send full chunks if tiny to avoid truncating key numbers) ----
    tiny_corpus = len(uniq) <= 3
    blocks = []
    for i, c in enumerate(uniq, start=1):
        snippet = (c["text"] or "") if tiny_corpus else (c["text"] or "")[:MAX_CHARS_PER_BLOCK]
        blocks.append(f"[C{i}] {snippet}\n-- Source: {c['title']} (chunk {c['idx']})")
    context = "\n\n".join(blocks)

    # ---- prompt shaping: allow ranges, insist on citing every claim ----
    style = choose_style(user_q)
    if style == "list":
        sys = (
            "Answer strictly from EVIDENCE CONTEXT. Provide a concise bulleted list. "
            "Cite every bullet with [C#]. If exact values are present, use them; "
            "if the evidence reports a range or qualitative bound (e.g., 'more than 70%'), report that faithfully. "
            "If evidence is insufficient, reply exactly: insufficient evidence."
        )
    elif style == "table":
        sys = (
            "Answer strictly from EVIDENCE CONTEXT. Provide a compact markdown table with appropriate columns. "
            "Cite each figure/bound in-table with [C#]. Use exact values when present; "
            "otherwise report ranges/bounds exactly as stated. "
            "If evidence is insufficient, reply exactly: insufficient evidence."
        )
    else:
        # sys = (
        #     "Answer strictly from the EVIDENCE CONTEXT. "
        #     "Use exact percentages and month/year when present. "
        #     "If the evidence reports a range or qualitative bound (e.g., 'more than 70%'), report that faithfully. "
        #     "When multiple time points exist, list each explicitly and include inline citations for each figure/bound "
        #     "(e.g., 53% in June 2024 [C1]; more than 70% by July 2025 [C2]). "
        #     "If evidence is insufficient, reply exactly: insufficient evidence."
        # )
      sys = (
            "Answer strictly from the EVIDENCE CONTEXT.\n"
            "Be concise: 1–2 sentences max.\n"
            "Use exact percentages and month/year if present; include inline [C#] for each figure.\n"
            "If the evidence reports a range/bound (e.g., 'more than 70%'), report it verbatim.\n"
            "If evidence is insufficient, reply exactly: insufficient evidence."
        )


    prompt = f"QUESTION:\n{user_q}\n\nEVIDENCE CONTEXT:\n{context}"

    # ---- generate ----
    answer = chat(
        [
            {"role": "system", "content": sys},
            {"role": "user",  "content": prompt},
        ],
    )
    trim = (answer or "").strip()
    if not re.search(r'[.!?]"?$', trim):
        # Ask for a one-shot continuation to finish the sentence—no new info
        continuation = chat(
            [
                {"role": "system", "content": sys},
                {"role": "user",  "content": prompt},
                {"role": "assistant", "content": trim},
                {"role": "user", "content": "Continue the previous answer. Finish the last sentence only; do not add new facts. Keep existing [C#] citations as-is."}
            ],
            max_tokens=120,  # small, just to close the thought
        )
        answer = (trim + " " + continuation.strip()).strip()


    # ---- evidence check (optional, controlled by env var) ----
    if ENABLE_EVIDENCE_CHECK:
        answer = evidence_check(answer, uniq)

    # ---- show only citations actually used in the answer (fallback to all uniq if none) ----
    used_nums = set(re.findall(r"\[C(\d+)\]", answer or ""))
    if used_nums:
        cits = []
        for i, c in enumerate(uniq, start=1):
            if str(i) in used_nums:
                cits.append({
                    "label": f"[C{i}]",
                    "title": c["title"],
                    "chunk_index": c["idx"],
                    "score_semantic": round(c["semantic"], 3),
                })
    else:
        cits = [{"label": f"[C{i}]",
                 "title": c["title"],
                 "chunk_index": c["idx"],
                 "score_semantic": round(c["semantic"], 3)}
                for i, c in enumerate(uniq, start=1)]

    return {"answer": answer, "citations": cits}

# --------- Debug endpoints ---------
@app.get("/debug/env")
def debug_env():
    return {"has_mistral_key": bool(os.getenv("MISTRAL_API_KEY"))}

@app.get("/debug/ping-emb")
def debug_ping_emb():
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        return {"ok": False, "error": "MISTRAL_API_KEY not set"}
    try:
        r = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "mistral-embed-v0.2", "input": ["hello"]},
            timeout=30,
        )
        return {"ok": r.status_code < 400, "status": r.status_code, "body": r.text[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/debug/preview-chunks")
async def preview_chunks(files: List[UploadFile] = File(...)):
    previews = []
    for f in files:
        data = await f.read()
        text = read_pdf_bytes(data)
        norm = re.sub(r"\s+", " ", text).strip()
        L = len(norm)
        eff_overlap = CHUNK_OVERLAP if CHUNK_OVERLAP < CHUNK_SIZE else max(0, CHUNK_SIZE // 5)
        step = max(1, CHUNK_SIZE - eff_overlap)
        expected = 0 if L == 0 else 1 + max(0, (L - 1) // step)
        previews.append({
            "file": f.filename,
            "text_length_chars": L,
            "CHUNK_SIZE": CHUNK_SIZE,
            "CHUNK_OVERLAP": CHUNK_OVERLAP,
            "effective_overlap_used": eff_overlap,
            "step": step,
            "expected_chunks": expected
        })
    return {"status": "ok", "preview": previews}
