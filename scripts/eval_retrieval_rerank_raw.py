from pathlib import Path
import json, random, statistics, urllib.parse

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---- must match your index ----
ROOT        = Path(__file__).resolve().parents[1]
PERSIST     = ROOT / "data" / "index" / "chroma"
COLLECTION  = "py_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SAMPLE      = 300
K_RETRIEVE  = 20   # get wider pool
K_FINAL     = 5    # report at k
QUERY_WORDS = 20
# --------------------------------

def norm_url(u: str) -> str:
    if not u: return ""
    pr = urllib.parse.urlparse(u)
    return urllib.parse.urlunparse(("https", pr.netloc, pr.path.rstrip("/"), "", "", ""))

def mrr_at_k(truth: str, urls: list[str]) -> float:
    for rank, u in enumerate(urls, start=1):
        if u == truth:
            return 1.0 / rank
    return 0.0

def main():
    print("PERSIST:", PERSIST)
    client = PersistentClient(path=str(PERSIST))
    col = client.get_collection(COLLECTION)

    # load sample chunks
    chunks_path = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
    chunks = [json.loads(l) for l in chunks_path.open("r", encoding="utf-8")]
    random.shuffle(chunks)
    sample = chunks[:SAMPLE]

    emb = SentenceTransformer(EMBED_MODEL)
    ce  = CrossEncoder(CE_MODEL)

    recalls, mrrs = [], []
    dbg = 0
    for rec in sample:
        truth = norm_url(rec.get("doc_url", ""))
        if not truth:
            continue
        q = " ".join(rec["text"].split()[:QUERY_WORDS])
        q_emb = emb.encode([q]).tolist()

        res = col.query(
            query_embeddings=q_emb,
            n_results=K_RETRIEVE,
            include=["documents", "metadatas", "distances"],
        )
        ids   = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        docs  = res.get("documents", [[]])[0]
        if not ids:
            continue

        scores = ce.predict([(q, d) for d in docs])
        order  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K_FINAL]
        urls   = [norm_url((metas[i] or {}).get("doc_url", "")) for i in order]

        recalls.append(int(truth in urls))
        mrrs.append(mrr_at_k(truth, urls))

        if dbg < 3:
            dbg += 1
            print("\n--- DEBUG ---")
            print("Truth:", truth)
            for rank, i in enumerate(order, 1):
                u = norm_url((metas[i] or {}).get("doc_url",""))
                print(f"[{rank}] {u}")

    n = len(recalls)
    print(f"\nEvaluated {n} samples (from {SAMPLE}), retrieve {K_RETRIEVE} → rerank → k={K_FINAL}")
    if n:
        print(f"Recall@{K_FINAL}: {sum(recalls)/n:.3f}")
        print(f"MRR@{K_FINAL}:    {statistics.mean(mrrs):.3f}")

if __name__ == "__main__":
    main()
