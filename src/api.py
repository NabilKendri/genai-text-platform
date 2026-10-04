import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from search import search
from graph import graph

app = FastAPI(title="GenAI Text Platform", version="1.0")

class SearchRequest(BaseModel):
    query: str 
    k: int = 5

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/search")
def search_docs(req: SearchRequest):
    start = time.perf_counter()
    results = search(req.query, k=req.k)
    return {
        "query": req.query,
        "results": [
            {"doc_id": d.metadata["doc_id"],"label": d.metadata["label"],
             "score": round(s, 3), "text": d.page_content}
            for d, s in results
        ],
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }

@app.post("/ask")
def ask(req: QueryRequest):
    start = time.perf_counter()
    result = graph.invoke({"query": req.query})
    return {
        "query": req.query,
        "awnser": result.get("awnser"),
        "sources": [d.metadata["doc_id"] for d in result ["docs"]],
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }

@app.post("/extract")
def extract(req: QueryRequest):
    start = time.perf_counter()
    result = graph.invoke({"query": f"extract: {req.query}"})
    if not result.get("extraction"):
        raise HTTPException(status_code=422, detail="Extraction failed")
    return {
        "query": req.query,
        "extraction": result["extraction"],
        "true_label": result["docs"][0].metadata["label"],
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }