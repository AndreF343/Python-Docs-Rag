from pathlib import Path
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PERSIST = ROOT / "data" / "index" / "chroma"
COLLECTION = "py_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match index build

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print("PERSIST:", PERSIST)
    client = PersistentClient(path=str(PERSIST))
    col = client.get_collection(COLLECTION)

    emb_model = SentenceTransformer(EMBED_MODEL)
    q_emb = emb_model.encode([args.query]).tolist()

    res = col.query(
        query_embeddings=q_emb,
        n_results=args.k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs  = res.get("documents", [[]])[0]

    if not ids:
        print("No results from Chroma.")
        return

    for i, (m, d) in enumerate(zip(metas, docs), 1):
        title = (m or {}).get("title", "(no title)")
        url   = (m or {}).get("doc_url", "")
        print(f"\n[{i}] {title}")
        print(f"URL: {url}")
        print((d or "").replace("\n", " ")[:300] + "...")
if __name__ == "__main__":
    main()
