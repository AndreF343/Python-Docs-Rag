from pathlib import Path
import json, random, statistics, urllib.parse
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
INDEX_DIR   = ROOT / "data" / "index" / "chroma"

SAMPLE = 300
K_RETRIEVE = 20   # take wider pool before rerank
K_FINAL    = 5    # report metrics at this k
QUERY_WORDS = 20
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def norm_url(u: str) -> str:
    if not u: return ""
    pr = urllib.parse.urlparse(u)
    return urllib.parse.urlunparse(("https", pr.netloc, pr.path.rstrip("/"), "", "", ""))

class SBertEmbeddings(Embeddings):
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
    def embed_documents(self, texts): return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):      return self.model.encode([text], convert_to_numpy=True)[0].tolist()

def mrr_at_k(truth: str, urls: list[str]) -> float:
    for rank, u in enumerate(urls, start=1):
        if u == truth: return 1.0 / rank
    return 0.0

def main():
    # load sample
    chunks = [json.loads(l) for l in CHUNKS_PATH.open("r", encoding="utf-8")]
    random.shuffle(chunks)
    sample = chunks[:SAMPLE]

    vs = Chroma(
        collection_name="py_docs",
        embedding_function=SBertEmbeddings(),
        persist_directory=str(INDEX_DIR),
        client_settings=Settings(anonymized_telemetry=False),
    )
    ce = CrossEncoder(CE_MODEL)

    recalls, mrrs = [], []
    dbg = 0
    for rec in sample:
        truth = norm_url(rec.get("doc_url",""))
        if not truth: continue
        q = " ".join(rec["text"].split()[:QUERY_WORDS])

        # retrieve K_RETRIEVE, then rerank to top K_FINAL
        docs = vs.similarity_search(q, k=K_RETRIEVE)
        if not docs: continue
        scores = ce.predict([(q, d.page_content) for d in docs])
        ranked = [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:K_FINAL]]

        urls = [norm_url((d.metadata or {}).get("doc_url","")) for d in ranked]
        recalls.append(int(truth in urls))
        mrrs.append(mrr_at_k(truth, urls))

        if dbg < 3:
            dbg += 1
            print("\n--- DEBUG ---")
            print("Truth:", truth)
            for i, d in enumerate(ranked, 1):
                print(f"[{i}] {norm_url((d.metadata or {}).get('doc_url',''))}")

    n = len(recalls)
    print(f"\nEvaluated {n} samples (from {SAMPLE}), final k={K_FINAL} (retrieve {K_RETRIEVE} then rerank)")
    if n:
        print(f"Recall@{K_FINAL}: {sum(recalls)/n:.3f}")
        print(f"MRR@{K_FINAL}:    {statistics.mean(mrrs):.3f}")

if __name__ == "__main__":
    main()
