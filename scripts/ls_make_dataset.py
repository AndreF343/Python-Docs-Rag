import json, random, re
from pathlib import Path
from langsmith import Client

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
DATASET_NAME = "py-docs-retrieval-smoke"

def norm_url(u: str) -> str:
    if not u:
        return ""
    # keep scheme+host+path only; strip fragments/query; coerce to https
    u = re.sub(r"#.*$", "", u.strip())  # strip fragments
    u = re.sub(r"\?.*$", "", u)         # strip query
    u = u.replace("http://", "https://")
    return u.rstrip("/")

def main():
    rows = []
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            u = norm_url(rec.get("doc_url", ""))
            if not u:
                continue
            # simple query = first N words
            q = " ".join(rec.get("text", "").split()[:20]).strip()
            if not q:
                continue
            rows.append({"inputs": {"query": q}, "outputs": {"doc_url": u}})

    # de-dup by (query, doc_url) and keep a manageable sample
    seen = set()
    uniq = []
    for r in rows:
        key = (r["inputs"]["query"], r["outputs"]["doc_url"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    random.shuffle(uniq)
    uniq = uniq[:50]  # small smoke set

    client = Client()
    # recreate dataset cleanly
    try:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(DATASET_NAME)
    except Exception:
        pass

    ds = client.create_dataset(DATASET_NAME, description="RAG retrieval smoke set with doc_url ground truth")
    client.create_examples(
        inputs=[r["inputs"] for r in uniq],
        outputs=[r["outputs"] for r in uniq],
        dataset_id=ds.id,
    )
    print(f"Dataset created: {DATASET_NAME}, size={len(uniq)}")

if __name__ == "__main__":
    main()
