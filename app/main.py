from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="RAG Mini")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def root():
    return '<meta http-equiv="refresh" content="0; url=/ui" />'

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>RAG Mini</title></head>
<body style="font-family:system-ui;margin:2rem">
  <h1>It works!</h1>
  <p>This is your placeholder UI. Next, paste in the full RAG UI & endpoints.</p>
</body></html>"""
