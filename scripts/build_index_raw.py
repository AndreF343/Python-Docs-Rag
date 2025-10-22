import json, itertools
from pathlib import Path
from typing import List, Dict, Any
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
PERSIST = ROOT / "data" / "index" / "chroma"
COLLECTION = "py_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match search/eval

BATCH = 512

def load_records(p: Path) -> List[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    PERSIST.mkdir(parents=True, exist_ok=True)
    recs = load_records(CHUNKS_PATH)
    if not recs:
        print("No records found:", CHUNKS_PATH); return

    # ids, texts, metadatas
    ids = [str(i) for i in range(len(recs))]
    texts = [r["text"] for r in recs]
    metas = [{"title": r.get("title",""), "doc_url": r.get("doc_url","")} for r in recs]

    # init chroma and (re)create collection
    client = PersistentClient(path=str(PERSIST))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})

    # embedder
    model = SentenceTransformer(EMBED_MODEL)

    # upsert in batches
    for start in range(0, len(texts), BATCH):
        end = start + BATCH
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]
        embs = model.encode(batch_texts, convert_to_numpy=True).tolist()
        col.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metas, embeddings=embs)
        print(f"Upserted {end}/{len(texts)}")

    print("Done. Count:", col.count(), "Persist:", PERSIST)

if __name__ == "__main__":
    main()
