import json, pickle, re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks" / "python_docs_chunks.jsonl"
OUTDIR = ROOT / "data" / "index" / "bm25"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-z0-9_#.\-]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())

def main():
    records: List[Dict[str, Any]] = []
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({
                "id": r.get("id"),
                "title": r.get("title") or "",
                "url": r.get("doc_url") or "",
                "text": r.get("text") or "",
            })

    # corpus for BM25
    corpus_tokens = [tokenize(r["text"]) for r in records]
    bm25 = BM25Okapi(corpus_tokens)

    # save BM25 + lightweight metadata
    with (OUTDIR / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25}, f)
    meta = [{"id": i, "title": r["title"], "url": r["url"]} for i, r in enumerate(records)]
    with (OUTDIR / "meta.pkl").open("wb") as f:
        pickle.dump({"meta": meta}, f)

    print(f"BM25 index written to: {OUTDIR}")

if __name__ == "__main__":
    main()
