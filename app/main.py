# app/main.py — Step 1: ingest + retrieve (no LLM generation yet)

import os, io, re, json, math, time, string, sqlite3, hashlib
from typing import List, Dict, Any
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
EMBED_MODEL_CANDIDATES = ["mistral-embed", "mistral-embed-v0.2", "mistral-embed-latest"]

DB_PATH = "rag.db"
DOC_DIR = "storage"
os.makedirs(DOC_DIR, exist_ok=True)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 8
FINAL_K = 4
SEMANTIC_WEIGHT = 0.6
SIM_THRESHOLD = 0.22

# --------- FastAPI ---------
app = FastAPI(title="RAG Mini")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
def health():
    return {"status":"ok", "has_key": bool(MISTRAL_API_KEY)}

@app.get("/", response_class=HTMLResponse)
def root():
    return '<meta http-equiv="refresh" content="0; url=/ui" />'

@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open("static/index.html","r",encoding="utf-8") as f:
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

def tf_counts(txt: str) -> Dict[str,int]:
    d = {}
    for t in tokenize(txt):
        d[t] = d.get(t, 0) + 1
    return d

def update_df(con, terms: List[str]):
    for t in set(terms):
        cur = con.execute("SELECT df FROM vocab WHERE term=?", (t,)).fetchone()
        if cur:
            con.execute("UPDATE vocab SET df=? WHERE term=?", (cur[0]+1, t))
        else:
            con.execute("INSERT INTO vocab(term, df) VALUES (?, ?)", (t, 1))

def all_chunk_count(con) -> int:
    return int(con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

def idf(con, term: str, total_chunks: int) -> float:
    row = con.execute("SELECT df FROM vocab WHERE term=?", (term,)).fetchone()
    df = row[0] if row else 0
    if df == 0: return 0.0
    return math.log((1 + total_chunks) / (1 + df)) + 1.0

# --------- Mistral API (embeddings) ---------
def _mistral_headers():
    if not MISTRAL_API_KEY:
        raise HTTPException(500, "Missing MISTRAL_API_KEY (set it in Space → Settings → Secrets).")
    return {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

def embed_texts(texts: List[str]) -> List[List[float]]:
    last_err = None
    for model in EMBED_MODEL_CANDIDATES:
        try:
            r = requests.post(
                f"{MISTRAL_BASE}/embeddings",
                headers=_mistral_headers(),
                json={"model": model, "input": texts},
                timeout=60
            )
            if r.status_code < 400:
                data = r.json()["data"]
                return [d["embedding"] for d in data]
            last_err = r.text
        except Exception as e:
            last_err = str(e)
    raise HTTPException(500, f"Mistral embeddings failed: {last_err}")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
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
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        chunk = text[i:i+size]
        last = chunk.rfind(". ")
        if last > size * 0.6:
            chunk = chunk[:last+1]
        out.append(chunk)
        i += max(1, len(chunk) - overlap)
    return [c for c in out if c.strip()]

# --------- Persist ---------
def save_document(filename: str, raw: bytes, chunks: List[str]):
    h = hashlib.sha256(raw).hexdigest()
    with db() as con:
        row = con.execute("SELECT id FROM documents WHERE doc_hash=?", (h,)).fetchone()
        if row:
            doc_id = row[0]
        else:
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
    return True

# --------- Hybrid search ---------
def hybrid_search(query: str, top_k=TOP_K) -> List[Dict[str,Any]]:
    q_emb = np.array(embed_texts([query])[0], dtype=np.float32)
    q_tf = tf_counts(query)

    with db() as con:
        total = all_chunk_count(con)
        rows = con.execute("""
            SELECT chunks.id, chunks.doc_id, chunks.idx, chunks.text, chunks.embedding, documents.title
            FROM chunks JOIN documents ON chunks.doc_id = documents.id
        """).fetchall()

    res = []
    # compute keyword-TFIDF score
    for cid, doc_id, idx, text, emb_json, title in rows:
        emb = np.array(json.loads(emb_json), dtype=np.float32)
        sem = cosine(q_emb, emb)

        tf = tf_counts(text)
        kw = 0.0
        # TF-IDF dot product (query vs chunk)
        with db() as con2:
            total = all_chunk_count(con2)
            for t, qcnt in q_tf.items():
                idf_q = idf(con2, t, total)
                kw += (qcnt * idf_q) * (tf.get(t,0) * idf_q)

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
        save_document(f.filename, data, chunks)
        out.append({"file": f.filename, "n_chunks": len(chunks)})
    return {"status":"ok", "ingested": out}

@app.post("/query")
def query(q: QueryIn):
    user_q = (q.query or "").strip()
    if not user_q:
        return {"answer":"Please type a question.", "citations":[]}

    # very light smalltalk guard
    if re.match(r"^\s*(hi|hello|hey)\b", user_q, re.I):
        return {"answer":"Hi! Upload PDFs, then ask about them.", "citations":[]}

    hits = hybrid_search(user_q, top_k=TOP_K)
    if not hits or max([h["semantic"] for h in hits]) < SIM_THRESHOLD:
        return {"answer":"insufficient evidence", "citations":[]}

    chosen = hits[:FINAL_K]
    # For Step 1 (no LLM), return top snippets as the “answer”
    snippet = "Here are the most relevant snippets:\n\n" + "\n\n".join(
        [f"[C{i+1}] {c['text'][:600]}..." for i, c in enumerate(chosen)]
    )
    cits = [{"label": f"[C{i+1}]", "title": c["title"], "chunk_index": c["idx"],
             "score_semantic": round(c["semantic"],3)} for i, c in enumerate(chosen)]
    return {"answer": snippet, "citations": cits}
