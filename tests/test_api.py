from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_search_returns_k_results():
    r = client.post("/search", json={"query": "oil prices", "k" : 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3