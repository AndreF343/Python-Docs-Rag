import json, re, time, collections
from urllib.parse import urljoin, urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SEEDS = [
    "https://docs.python.org/3/library/index.html",
    "https://docs.python.org/3/reference/index.html",
    # Uncomment to include How-To guides:
    "https://docs.python.org/3/howto/index.html",
]

BASE_PREFIX = "https://docs.python.org/3/"
ALLOWED_PREFIXES = (
    "https://docs.python.org/3/library/",
    "https://docs.python.org/3/reference/",
    "https://docs.python.org/3/howto/",
)

# Heuristic skips: noisy or non-content pages
SKIP_SUFFIXES = (
    "genindex.html",
    "search.html",
    "py-modindex.html",
    "glossary.html",
    "whatsnew/index.html",
)
HEADERS = {"User-Agent": "DocsCollector/0.1 (personal project)"}
S = requests.Session()

OUT_DIR = Path("../data")
RAW_DIR = OUT_DIR / "raw_html"
CLEAN_DIR = OUT_DIR / "clean"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_PATH = CLEAN_DIR / "python_docs_selected.jsonl"

MAX_PAGES = 500         # safety cap
REQUEST_DELAY = 0.2     # seconds between requests

def should_visit(url: str) -> bool:
    if not url.startswith(BASE_PREFIX):
        return False
    if not url.startswith(ALLOWED_PREFIXES):
        return False
    if url.endswith(SKIP_SUFFIXES):
        return False
    # Drop anchors and query
    return True

def normalize_url(url: str) -> str:
    # Strip fragments and query
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl()

def extract_links(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        href = normalize_url(href)
        if should_visit(href):
            links.append(href)
    return links

def extract_main_text(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    # Prefer role="main" or Sphinx .document container
    main = soup.find(attrs={"role": "main"}) or soup.select_one("div.document")
    text = (main or soup).get_text(separator="\n", strip=True)

    # Collapse whitespace, drop empties
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    cleaned = "\n".join(lines)

    title = soup.title.get_text(strip=True) if soup.title else url
    return title, cleaned

def save_raw(url: str, html: str) -> None:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url.strip("/"))[:180]
    (RAW_DIR / f"{safe}.html").write_text(html, encoding="utf-8")

def save_clean(url: str, title: str, text: str) -> None:
    with CLEAN_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False) + "\n")

def crawl(limit: int = MAX_PAGES):
    seen = set()
    q = collections.deque()

    for seed in SEEDS:
        q.append(seed)
        seen.add(seed)

    pbar = tqdm(total=limit, desc="Crawling")
    saved = 0

    while q and saved < limit:
        url = q.popleft()
        try:
            resp = S.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            html = resp.text

            # Save raw + clean
            save_raw(url, html)
            title, text = extract_main_text(html, url)
            if len(text.split()) >= 80:  # skip trivial pages
                save_clean(url, title, text)
                saved += 1
                pbar.update(1)

            # Enqueue children
            for href in extract_links(url, html):
                if href not in seen:
                    seen.add(href)
                    q.append(href)

            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"[warn] {url}: {e}")
            continue

    pbar.close()
    print(f"Done. Saved {saved} pages to {CLEAN_PATH}")

if __name__ == "__main__":
    crawl()
