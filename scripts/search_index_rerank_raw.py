from pathlib import Path
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---- CONFIG: must match how you built the index ----
ROOT = Path(__file__).resolve().parents[1]
PERSIST = ROOT / "data" / "index" / "chroma"
COLLECTION = "py_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # must match build_index.py
CE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_RETRIEVE  = 12
K_FINAL     = 5
# ----------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=K_RETRIEVE)
    ap.add_argument("--top", type=int, default=K_FINAL)
    args = ap.parse_args()

    print("PERSIST:", PERSIST)
    client = PersistentClient(path=str(PERSIST))
    col = client.get_collection(COLLECTION)

    # embed query (same model used to build index)
    emb_model = SentenceTransformer(EMBED_MODEL)
    q_emb = emb_model.encode([args.query]).tolist()

    # retrieve
    res = col.query(
        query_embeddings=q_emb,
        n_results=args.k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    if not ids:
        print("No results from Chroma.")
        return

    # rerank
    ce = CrossEncoder(CE_MODEL)
    scores = ce.predict([(args.query, doc) for doc in docs])
    ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)[:args.top]

    for i, (id_, doc, meta, score) in enumerate(ranked, 1):
        title = (meta or {}).get("title", "(no title)")
        url   = (meta or {}).get("doc_url", "")
        print(f"\n[{i}] score={score:.3f}  {title}")
        print(f"URL: {url}")
        one = doc.replace("\n", " ")
        print(one[:400] + ("..." if len(one) > 400 else ""))

if __name__ == "__main__":
    main()
