from pathlib import Path
import json, random, statistics, urllib.parse, time, os
import mlflow

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---- must match your index ----
ROOT        = Path(__file__).resolve().parents[1]
PERSIST     = ROOT / "data" / "index" / "chroma"
COLLECTION  = "py_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SAMPLE      = 300
K_RETRIEVE  = 10
K_FINAL     = 5
QUERY_WORDS = 20
EXPERIMENT  = "py-docs-rag"

K_RETRIEVE = int(os.getenv("K_RETRIEVE", K_RETRIEVE))
K_FINAL = int(os.getenv("K_FINAL", K_FINAL))
EMBED_MODEL = os.getenv("EMBED_MODEL", EMBED_MODEL)
CE_MODEL = os.getenv("CROSS_ENCODER_MODEL", CE_MODEL)
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
    # 1) Set local file store (no server required)
    mlruns_dir = ROOT / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)

    # ✅ Proper file:// URI on Windows (file:///C:/... with forward slashes)
    mlruns_uri = mlruns_dir.resolve().as_uri()

    override_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if override_uri:
        mlflow.set_tracking_uri(override_uri)
    else:
        mlruns_dir = ROOT / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri((mlruns_dir.resolve()).as_uri())

    mlflow.set_tracking_uri(mlruns_uri)

    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="eval_rerank_raw") as run:
        # Log params describing this config
        mlflow.log_params({
            "embed_model": EMBED_MODEL,
            "cross_encoder": CE_MODEL,
            "k_retrieve": K_RETRIEVE,
            "k_final": K_FINAL,
            "sample": SAMPLE,
            "query_words": QUERY_WORDS,
            "index_dir": str(PERSIST),
            "collection": COLLECTION,
        })

        client = PersistentClient(path=str(PERSIST))
        col = client.get_collection(COLLECTION)

        # Load sample
        chunks_path = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
        chunks = [json.loads(l) for l in chunks_path.open("r", encoding="utf-8")]
        random.shuffle(chunks)
        sample = chunks[:SAMPLE]

        emb = SentenceTransformer(EMBED_MODEL)
        ce  = CrossEncoder(CE_MODEL)

        recalls, mrrs = [], []
        debug_samples = []
        t0 = time.time()

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
            urls   = [norm_url((metas[i] or {}).get("doc_url","")) for i in order]

            recalls.append(int(truth in urls))
            mrrs.append(mrr_at_k(truth, urls))

            if dbg < 3:
                dbg += 1
                debug_samples.append({
                    "query": q,
                    "truth": truth,
                    "top_urls": urls
                })

        elapsed = time.time() - t0
        n = len(recalls)
        recall_k = (sum(recalls)/n) if n else 0.0
        mrr_k = (statistics.mean(mrrs)) if n else 0.0

        # Log metrics
        mlflow.log_metrics({
            f"Recall_at_{K_FINAL}": recall_k,
            f"MRR_at_{K_FINAL}": mrr_k,
            "evaluated": n,
            "elapsed_sec": elapsed
        })

        # Save debug samples as an artifact
        artifacts_dir = ROOT / "data" / "analysis"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        dbg_path = artifacts_dir / "eval_debug_samples.json"
        dbg_path.write_text(json.dumps(debug_samples, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(dbg_path))

        print(f"Evaluated {n} samples (from {SAMPLE}), retrieve {K_RETRIEVE} → rerank → k={K_FINAL}")
        print(f"Recall@{K_FINAL}: {recall_k:.3f}")
        print(f"MRR@{K_FINAL}:    {mrr_k:.3f}")
        print("Run ID:", run.info.run_id)
        print("Tracking dir:", mlruns_dir)

if __name__ == "__main__":
    main()
