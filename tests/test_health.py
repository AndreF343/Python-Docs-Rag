import pytest, httpx

@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/health"])
def test_health(url):
    r = httpx.get(url, timeout=30)
    assert r.status_code == 200
    for k in ["status","index_dir","collection","embed_model","cross_encoder"]:
        assert k in r.json()
