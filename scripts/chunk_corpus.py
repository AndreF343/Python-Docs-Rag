import json, hashlib, os
from pathlib import Path

CLEAN = Path("../data/clean/python_docs_selected.jsonl")
OUT_DIR = Path("../data/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "python_docs_chunks.jsonl"

# Tweak these:
CHUNK_SIZE = 220      # words per chunk (≈ 1500–2000 chars for docs prose)
CHUNK_OVERLAP = 40    # words overlapped between consecutive chunks
MAX_CHUNKS_PER_DOC = 999  # safety

def chunk_words(words, size, overlap):
    assert size > 0 and overlap >= 0 and overlap < size
    i = 0
    while i < len(words):
        yield words[i:i+size]
        i += size - overlap

def stable_doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]

def main():
    if not CLEAN.exists():
        raise SystemExit(f"Missing input: {CLEAN}")
    n_pages = 0
    n_chunks = 0
    with CLEAN.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            url = rec["url"]
            title = rec.get("title","")
            text = rec.get("text","")
            words = text.split()
            if not words:
                continue
            base_id = stable_doc_id(url)
            for j, wchunk in enumerate(chunk_words(words, CHUNK_SIZE, CHUNK_OVERLAP), start=1):
                if j > MAX_CHUNKS_PER_DOC:
                    break
                chunk_text = " ".join(wchunk).strip()
                if len(chunk_text) < 50:  # skip ultra-short
                    continue
                chunk_id = f"{base_id}_{j:04d}"
                out = {
                    "id": chunk_id,
                    "doc_url": url,
                    "title": title,
                    "chunk_index": j,
                    "text": chunk_text,
                    "meta": {
                        "chunk_size_words": CHUNK_SIZE,
                        "chunk_overlap_words": CHUNK_OVERLAP
                    }
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_pages += 1
    print(f"Chunked {n_pages} pages into {n_chunks} chunks → {OUT_PATH}")

if __name__ == "__main__":
    main()
