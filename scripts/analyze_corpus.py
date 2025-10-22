import json, statistics, os
from pathlib import Path
from collections import Counter, defaultdict

CLEAN = Path("../data/clean/python_docs_selected.jsonl")
OUT_DIR = Path("../data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = OUT_DIR / "stats.json"

def word_count(text: str) -> int:
    return len(text.split())

def section_from_url(url: str) -> str:
    # crude bucket: /library/, /reference/, /howto/
    for sec in ("library", "reference", "howto"):
        if f"/{sec}/" in url:
            return sec
    return "other"

def main():
    if not CLEAN.exists():
        raise SystemExit(f"Missing input: {CLEAN}")
    n_docs = 0
    words = []
    by_section = defaultdict(list)
    sample_titles = []
    with CLEAN.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            n_docs += 1
            wc = word_count(rec["text"])
            words.append(wc)
            by_section[section_from_url(rec["url"])].append(wc)
            if len(sample_titles) < 10:
                sample_titles.append(rec["title"])

    stats = {
        "n_docs": n_docs,
        "total_words": int(sum(words)),
        "avg_words_per_doc": float(statistics.mean(words)) if words else 0.0,
        "median_words_per_doc": float(statistics.median(words)) if words else 0.0,
        "p90_words_per_doc": float(sorted(words)[int(0.9*len(words))-1]) if words else 0.0,
        "sections": {
            sec: {
                "count": len(lst),
                "avg_words": float(statistics.mean(lst)) if lst else 0.0
            } for sec, lst in by_section.items()
        },
        "sample_titles": sample_titles
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {STATS_PATH}")
    print(json.dumps(stats, indent=2)[:800], "...\n")

if __name__ == "__main__":
    main()
