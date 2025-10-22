import sys, json, textwrap, requests

API = "http://127.0.0.1:8000"  # change if needed

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo_cli.py \"your question here\"")
        sys.exit(1)
    query = sys.argv[1]
    payload = {"query": query, "k_retrieve": 20, "k_final": 5}
    r = requests.post(f"{API}/search", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    print(f"\nQ: {query}\n")
    if not results:
        print("No results.")
        return
    for i, res in enumerate(results, 1):
        title = res.get("title") or "(no title)"
        url = res.get("url") or ""
        score = res.get("score")
        snippet = (res.get("snippet") or "").strip()
        print(f"{i}. {title}  [score={score:.3f}]" if score is not None else f"{i}. {title}")
        if url:
            print(f"   {url}")
        if snippet:
            wrap = textwrap.fill(snippet, width=96, subsequent_indent="   ")
            print("   " + wrap)
        print()
    print("(Tip: open the first URL to verify relevance.)")

if __name__ == "__main__":
    main()
