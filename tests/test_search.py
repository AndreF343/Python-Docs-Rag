import httpx, json
def test_search():
    payload = {"query":"create a virtual environment","k_retrieve":10,"k_final":5}
    r = httpx.post("http://127.0.0.1:8000/search", json=payload, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert data["results"], "no results"
    assert "url" in data["results"][0]
