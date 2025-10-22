from pathlib import Path
import json, sys

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
INDEX_DIR   = ROOT / "data" / "index" / "chroma"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

class SBertEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

def main():
    if not CHUNKS_PATH.exists():
        print(f"Missing input: {CHUNKS_PATH}", file=sys.stderr); sys.exit(1)

    embed = SBertEmbeddings()
    vs = Chroma(
        collection_name="py_docs",
        embedding_function=embed,
        persist_directory=str(INDEX_DIR),
        client_settings=Settings(anonymized_telemetry=False),
    )

    batch_size = 1000
    buf_texts, buf_metas, buf_ids = [], [], []
    total = 0

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            # Build a consistent metadata dict no matter how the chunk JSON is shaped
            meta = rec.get("metadata") or {}
            # Merge in the fields our pipeline expects
            meta = {
                "doc_url": rec.get("doc_url", meta.get("doc_url", "")),
                "title": rec.get("title", meta.get("title", "")),
                "chunk_index": rec.get("chunk_index", meta.get("chunk_index", 0)),
                **(rec.get("meta") or {}),   # ← your chunk-size/overlap info
            }

            buf_texts.append(rec["text"])
            buf_metas.append(meta)
            buf_ids.append(rec["id"])

            if len(buf_texts) >= batch_size:
                vs.add_texts(texts=buf_texts, metadatas=buf_metas, ids=buf_ids)
                total += len(buf_texts)
                print(f"Indexed {total} chunks…")
                buf_texts, buf_metas, buf_ids = [], [], []

    if buf_texts:
        vs.add_texts(texts=buf_texts, metadatas=buf_metas, ids=buf_ids)
        total += len(buf_texts)
        print(f"Indexed {total} chunks…")

    #vs.persist()
    print(f"Done. Index at {INDEX_DIR}")


if __name__ == "__main__":
    main()
