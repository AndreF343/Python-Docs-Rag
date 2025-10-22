import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma  # keep same wrapper you used to build
from chromadb.config import Settings

# Paths
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index" / "chroma"

# Embedding wrapper (same model as index)
class SBertEmbeddings(Embeddings):
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
    def embed_documents(self, texts): return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):      return self.model.encode([text], convert_to_numpy=True)[0].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="your query")
    ap.add_argument("--k", type=int, default=12, help="initial retrieval size")
    ap.add_argument("--top", type=int, default=5, help="final results after rerank")
    ap.add_argument("--ce", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="cross-encoder model")
    args = ap.parse_args()

    # 1) retrieve
    vs = Chroma(
        collection_name="py_docs",
        embedding_function=SBertEmbeddings(),
        persist_directory=str(INDEX_DIR),
        client_settings=Settings(anonymized_telemetry=False),
    )
    docs = vs.similarity_search(args.query, k=args.k)

    if not docs:
        print("No results."); return

    # 2) re-rank
    ce = CrossEncoder(args.ce)
    pairs = [(args.query, d.page_content) for d in docs]
    scores = ce.predict(pairs)  # higher is better
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:args.top]

    for i, (d, s) in enumerate(ranked, 1):
        meta = d.metadata or {}
        print(f"\n[{i}] score={s:.3f}  {meta.get('title','(no title)')}")
        print(f"URL: {meta.get('doc_url','')}")
        txt = d.page_content.replace("\n"," ")
        print(txt[:400] + ("..." if len(txt) > 400 else ""))

if __name__ == "__main__":
    main()
