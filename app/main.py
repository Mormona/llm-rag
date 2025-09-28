# app/main.py — Phase A (stubs only)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="RAG Mini")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def root():
    return '<meta http-equiv="refresh" content="0; url=/ui" />'

@app.get("/ui", response_class=HTMLResponse)
def ui():
    # keep your current placeholder page
    return """<!doctype html><html><head><meta charset="utf-8"><title>RAG Mini</title></head>
<body style="font-family:system-ui;margin:2rem">
  <h1>It works!</h1>
  <p>Stub endpoints are live. Next, we’ll implement real ingest+query.</p>
</body></html>"""

# ---------- RAG endpoints (stubs) ----------
class QueryIn(BaseModel):
    query: str

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    # Just prove uploads reach the server.
    names = [f.filename for f in files]
    return {"status": "ok", "received": names, "note": "RAG logic not implemented yet"}

@app.post("/query")
def query(q: QueryIn):
    # Just echo back. UI can call this today.
    return {"answer": f"(stub) you asked: {q.query}", "citations": []}
