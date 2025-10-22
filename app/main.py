import os, re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import requests as _requests
from .settings import MLFLOW_MODEL_URI
import json, pickle
import numpy as np

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from .settings import (
    INDEX_DIR, COLLECTION_NAME,
    EMBED_MODEL, CROSS_ENCODER_MODEL,
    K_RETRIEVE_DEFAULT, K_FINAL_DEFAULT,
    OLLAMA_BASE, OLLAMA_MODEL, MAX_TOKENS_ANSWER, TEMPERATURE, MIN_SCORE,
    HYBRID, BM25_DIR, VEC_CAND, BM25_CAND, RRF_K,
    COMPRESS, COMPRESS_MODE, SENTS_PER_DOC, MIN_SENT_SCORE, MAX_CONTEXT_CHARS
)

import logging, json, time, uuid
from fastapi import Request

from langsmith import Client, traceable

MIN_SCORE = float(os.getenv("MIN_SCORE", "0.15"))  # tweak in MLflow sweeps later

client_ls = Client()  # uses env credentials

logger = logging.getLogger("api")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.handlers = [handler]
logger.setLevel(logging.INFO)

app = FastAPI(title="Python Docs RAG API (retrieval + rerank)")

def _norm_url(u: str) -> str:
    """Normalize URLs for display & eval: strip query/fragment, https, no trailing slash."""
    if not u:
        return ""
    u = u.strip()
    u = re.sub(r"[?#].*$", "", u)     # strip ?query and #fragment
    u = u.replace("http://", "https://")
    return u.rstrip("/")

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(json.dumps({
        "event":"http_request",
        "path": request.url.path,
        "status": response.status_code,
        "elapsed_ms": int(elapsed*1000),
        "request_id": rid,
    }))
    response.headers["x-request-id"] = rid
    return response



# --------- Startup: load models & collection once ----------
_client = None
_collection = None
_embedder = None
_cross = None
bm25 = None
bm25_meta = None

@app.on_event("startup")
def _startup():
    global _client, _collection, _embedder, _cross
    global bm25, bm25_meta
    # If provided, load config from MLflow Registry
    cfg = None
    if MLFLOW_MODEL_URI:
        import mlflow.pyfunc
        m = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        cfg = m.predict(None)  # returns dict we logged

    # fall back to env/defaults if not set
    index_dir = cfg["index_dir"] if cfg else str(INDEX_DIR)
    collection = cfg["collection"] if cfg else COLLECTION_NAME
    embed_model = cfg["embed_model"] if cfg else EMBED_MODEL
    cross_model = cfg["cross_encoder"] if cfg else CROSS_ENCODER_MODEL

    _client = PersistentClient(path=index_dir)
    _collection = _client.get_collection(collection)
    _embedder = SentenceTransformer(embed_model)
    _cross = CrossEncoder(cross_model)

    # --- optional BM25 (hybrid mode) ---
    bm25 = None
    bm25_meta = None
    if HYBRID:
        try:
            with (BM25_DIR / "bm25.pkl").open("rb") as f:
                bm25 = pickle.load(f)["bm25"]
            with (BM25_DIR / "meta.pkl").open("rb") as f:
                bm25_meta = pickle.load(f)["meta"]
            print(f"[HYBRID] BM25 loaded with {len(bm25_meta)} docs")
        except Exception as e:
            print(f"[HYBRID] BM25 not available: {e}. Falling back to vector-only.")
            bm25 = None

# --------- Schemas ----------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    k_retrieve: int = Field(K_RETRIEVE_DEFAULT, ge=1, le=100)
    k_final: int = Field(K_FINAL_DEFAULT, ge=1, le=50)

class Source(BaseModel):
    title: Optional[str] = ""
    url: Optional[str] = ""
    snippet: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[Source]
    answerable: bool = True                 # NEW
    # k_retrieve: Optional[int] = None        # optional, handy for debugging
    # k_final: Optional[int] = None           # optional, handy for debugging

# --------- Routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_dir": str(INDEX_DIR),
        "collection": COLLECTION_NAME,
        "count": _collection.count() if _collection else 0,
        "embed_model": EMBED_MODEL,
        "cross_encoder": CROSS_ENCODER_MODEL,
    }

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    # # 1) embed query
    q_emb = _embed(req.query)
    res = _collection.query(
        query_embeddings=q_emb,
        n_results=max(req.k_retrieve, VEC_CAND),   # more vec candidates when hybrid
        include=["documents", "metadatas", "distances"],
    )

    vec_docs = []
    ids   = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    if not ids:
        return SearchResponse(query=req.query, results=[], answerable=False)

    for doc, meta in zip(docs, metas):
        vec_docs.append({
            "title": meta.get("title") or "",
            "url": _norm_url(meta.get("doc_url") or ""),
            "snippet": (doc or "").replace("\n", " "),
        })

    # 2) optional BM25 and RRF fuse
    if HYBRID and bm25 and bm25_meta:
        # BM25 top N (by index position in bm25_meta)
        import math
        # scores -> top IDs
        bm25_scores = bm25.get_scores(_tok(req.query))
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda t: t[1], reverse=True)[:BM25_CAND]
        bm25_ids = [i for (i, _) in bm25_ranked]

        # map vector docs to BM25 ids by URL (robust)
        url_to_bm25 = {m["url"]: i for i, m in enumerate(bm25_meta) if m.get("url")}
        vec_ids = [url_to_bm25[d["url"]] for d in vec_docs if d.get("url") in url_to_bm25]

        # fuse to get candidate bm25 IDs
        ranks = {}
        for r, did in enumerate(vec_ids, 1):
            ranks[did] = ranks.get(did, 0.0) + 1.0 / (RRF_K + r)
        for r, did in enumerate(bm25_ids, 1):
            ranks[did] = ranks.get(did, 0.0) + 1.0 / (RRF_K + r)
        fused_ids = [did for (did, _) in sorted(ranks.items(), key=lambda t: t[1], reverse=True)]
        fused_ids = fused_ids[:max(req.k_retrieve, VEC_CAND, BM25_CAND)]

        # build candidate docs from BM25 metadata; reuse snippets from vec_docs when possible
        url_to_snip = {d["url"]: d["snippet"] for d in vec_docs}
        base = []
        for did in fused_ids:
            m = bm25_meta[did]
            url = _norm_url(m.get("url") or "")
            base.append({
                "title": m.get("title") or "",
                "url": url,
                "snippet": url_to_snip.get(url, ""),  # may be empty if only from BM25
            })
    else:
        # vector-only candidates
        base = vec_docs

    # (Optional) drop any empty-snippet candidates (improves rerank stability)
    base = [d for d in base if d.get("snippet")]

    # 3) rerank with cross-encoder
    pairs = [(req.query, d["snippet"]) for d in base]
    scores = _rerank(req.query, [d["snippet"] for d in base])

    # 4) take top k_final
    order = sorted(range(len(base)), key=lambda i: scores[i], reverse=True)[:req.k_final]
    results = []
    for i in order:
        d = base[i]
        results.append(Source(
            title=d["title"],
            url=d["url"],
            snippet=d["snippet"][:600],
            score=float(scores[i]),
        ))

    answerable = bool(results) and (results[0].score is not None) and (results[0].score >= MIN_SCORE)
    return SearchResponse(query=req.query, results=results, answerable=answerable)



class AnswerRequest(BaseModel):
    query: str
    k_retrieve: int = 20
    k_final: int = 5
    max_tokens: int | None = None
    temperature: float | None = None

class AnswerResponse(BaseModel):
    query: str
    answerable: bool
    answer: str | None
    sources: list[dict]  # [{title, url, score}]
    model: str

def _format_context_for_llm(query: str, docs: list[dict]) -> str:
    """Create a compact, numbered context block for the LLM with source tags."""
    # docs: [{"title","url","snippet","score"}, ...] — already reranked
    lines = [f"Question: {query}", "", "Context (numbered snippets):"]
    for i, d in enumerate(docs, 1):
        title = d.get("title") or ""
        url = d.get("url") or ""
        snip = (d.get("snippet") or "").strip()
        lines.append(f"[{i}] {title}\nURL: {url}\n{snip}\n")
    return "\n".join(lines)

def _build_prompt(query: str, docs: list[dict]) -> str:
    ctx = _format_context_for_llm(query, docs)
    instr = (
        "You are a helpful assistant. Answer the question *only* using the Context.\n"
        "If the context does not contain the answer, say you don't know.\n"
        "Cite sources inline like [1], [2] matching the numbered snippets, and include a brief Sources list at the end with full URLs.\n"
        "Be concise (3–6 sentences)."
    )
    return f"{instr}\n\n{ctx}\n\nNow, provide the answer with citations."

def _ollama_generate(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    # Using Ollama /api/generate (simple; /api/chat also works)
    resp = _requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # data: { "response": "...", "done": true, ... }
    return data.get("response", "").strip()

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    # 1) retrieve + rerank (reuse your existing pipeline)
    =q_emb = _embed(req.query)
    res = _collection.query(
        query_embeddings=q_emb,
        n_results=max(req.k_retrieve, VEC_CAND),   # use bigger pool for hybrid fusion
        include=["documents", "metadatas", "distances"],
    )
    ids   = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    if not ids:
        # return an empty-answers response in this handler
        # (in /search you’ll return SearchResponse; in /answer handle accordingly)
        ...
    # Build vector docs
    vec_docs = []
    for doc, meta in zip(docs, metas):
        vec_docs.append({
            "title": meta.get("title") or "",
            "url": _norm_url(meta.get("doc_url") or ""),
            "snippet": (doc or "").replace("\n", " "),
        })

    # If HYBRID is on and BM25 is available → RRF fuse vec + bm25
    if HYBRID and bm25 and bm25_meta:
        # BM25 top-N
        bm25_scores = bm25.get_scores(_tok(req.query))
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda t: t[1], reverse=True)[:BM25_CAND]
        bm25_ids = [i for (i, _) in bm25_ranked]

        # map vector docs to BM25 indices by URL (robust)
        url_to_bm25 = {m["url"]: i for i, m in enumerate(bm25_meta) if m.get("url")}
        vec_ids = [url_to_bm25[d["url"]] for d in vec_docs if d.get("url") in url_to_bm25]

        # RRF fuse
        ranks = {}
        for r, did in enumerate(vec_ids, 1):
            ranks[did] = ranks.get(did, 0.0) + 1.0 / (RRF_K + r)
        for r, did in enumerate(bm25_ids, 1):
            ranks[did] = ranks.get(did, 0.0) + 1.0 / (RRF_K + r)
        fused_ids = [did for (did, _) in sorted(ranks.items(), key=lambda t: t[1], reverse=True)]
        fused_ids = fused_ids[:max(req.k_retrieve, VEC_CAND, BM25_CAND)]

        # Build base candidates from BM25 metadata; reuse snippets from vec when possible
        url_to_snip = {d["url"]: d["snippet"] for d in vec_docs}
        base = []
        for did in fused_ids:
            m = bm25_meta[did]
            url = _norm_url(m.get("url") or "")
            snip = url_to_snip.get(url, "")
            if not snip:
                # optional: keep empty → cross-encoder will down-rank it
                # or you can skip if you prefer:
                #   continue
                pass
            base.append({
                "title": m.get("title") or "",
                "url": url,
                "snippet": snip,
            })
    else:
        # vector-only candidates
        base = vec_docs

    # (optional) drop empty snippets to stabilize rerank
    base = [d for d in base if d.get("snippet")]


    # 2) rerank
    pairs = [(req.query, d["snippet"]) for d in base]
    scores = _cross.predict(pairs).tolist()
    for i, s in enumerate(scores):
        base[i]["score"] = float(s)
    base.sort(key=lambda d: d["score"] if d["score"] is not None else -1e9, reverse=True)

    # 3) final cut & answerability
    topk = base[:req.k_final]
    top_score = topk[0]["score"] if topk and topk[0]["score"] is not None else 0.0
    answerable = top_score >= MIN_SCORE

    # Always return sources; only call LLM if answerable
    sources_dedup = []
    seen = set()
    for d in topk:
        u = (d.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            sources_dedup.append({"title": d.get("title") or "", "url": u, "score": d.get("score")})

    if not answerable:
        return AnswerResponse(
            query=req.query,
            answerable=False,
            answer=None,
            sources=sources_dedup,
            model=OLLAMA_MODEL,
        )

    # 4) Prompt & call Ollama
    # prompt = _build_prompt(req.query, topk)
    # === NEW: compress context before prompting the LLM ===
    if COMPRESS:
        ctx_text, ctx_sources = _build_compressed_context(
            req.query, topk,
            mode=COMPRESS_MODE,
            per_doc=SENTS_PER_DOC,
            min_score=MIN_SENT_SCORE,
            max_chars=MAX_CONTEXT_CHARS,
        )
        # prefer compressed sources order (subset of topk)
        sources_dedup = ctx_sources or sources_dedup
        prompt = (
            "You are a helpful assistant. Answer the question ONLY using the Context.\n"
            "If the context does not contain the answer, say you don't know.\n"
            "Cite sources as [n] where n matches the numbered snippets, and include a brief Sources list at the end with full URLs.\n"
            "Be concise (3–6 sentences).\n\n"
            f"{ctx_text}\n\nNow, provide the answer with citations."
        )
    else:
        # fallback to original (uncompressed) prompt builder you had
        prompt = _build_prompt(req.query, topk)

    model = OLLAMA_MODEL if not req.temperature else OLLAMA_MODEL  # model fixed; temp & tokens can vary
    max_tokens = req.max_tokens or MAX_TOKENS_ANSWER
    temperature = req.temperature if req.temperature is not None else TEMPERATURE

    try:
        text = _ollama_generate(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        # If the LLM fails, degrade gracefully to retrieval-only response
        text = f"(Generation failed: {e!s})\n\nTop sources:\n" + "\n".join([f"- {s['url']}" for s in sources_dedup])

    return AnswerResponse(
        query=req.query,
        answerable=True,
        answer=text,
        sources=sources_dedup,
        model=model,
    )


@traceable(name="embed_query")
def _embed(q: str):
    return _embedder.encode([q]).tolist()

@traceable(name="retrieve")
def _retrieve(q_emb, k):
    return _collection.query(query_embeddings=q_emb, n_results=k, include=["documents","metadatas","distances"])

@traceable(name="rerank")
def _rerank(query, docs):
    pairs = [(query, d) for d in docs]
    return _cross.predict(pairs)

TOKEN_RE = re.compile(r"[A-Za-z0-9_#.\-]+")

def _tok(s: str):
    return TOKEN_RE.findall((s or "").lower())

def _bm25_top_indices(query: str, k: int):
    # returns list of (doc_idx, score) descending
    if not bm25 or not bm25_meta:
        return []
    scores = bm25.get_scores(_tok(query))
    top = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)[:k]
    return top  # (idx, score)

def _rrf_fuse(vec_ids: list[int], bm25_ids: list[int], k: int) -> list[int]:
    """
    Reciprocal Rank Fusion over two ranked lists.
    rrf_score = sum(1 / (RRF_K + rank))
    """
    ranks = {}
    for rank, doc_id in enumerate(vec_ids, start=1):
        ranks.setdefault(doc_id, 0.0)
        ranks[doc_id] += 1.0 / (RRF_K + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        ranks.setdefault(doc_id, 0.0)
        ranks[doc_id] += 1.0 / (RRF_K + rank)
    # sort by fused score desc
    fused = sorted(ranks.items(), key=lambda t: t[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:k]]

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(])")

def _split_sentences(text: str) -> list[str]:
    # conservative split; keeps brackets and code-ish punctuation
    text = (text or "").strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    # normalize whitespace & drop empties/very short stubs
    clean = []
    for s in sents:
        s = " ".join(s.split())
        if len(s) >= 20:
            clean.append(s)
    return clean

def _top_sentences_embed(query: str, snippet: str, k: int, min_score: float) -> list[tuple[str, float]]:
    """Score sentences by cosine(q_emb, sent_emb)."""
    sents = _split_sentences(snippet)
    if not sents:
        return []
    q_emb = _embedder.encode([query])[0]
    s_embs = _embedder.encode(sents)
    # cosine similarity
    q = np.asarray(q_emb, dtype="float32")
    s = np.asarray(s_embs, dtype="float32")
    qn = q / (np.linalg.norm(q) + 1e-9)
    sn = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
    sims = sn.dot(qn)
    pairs = [(sents[i], float(sims[i])) for i in range(len(sents))]
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [(sent, score) for sent, score in pairs[:k] if score >= min_score]

def _top_sentences_crossenc(query: str, snippet: str, k: int, min_score: float) -> list[tuple[str, float]]:
    """Score sentences with the cross-encoder (slower, sharper)."""
    sents = _split_sentences(snippet)
    if not sents:
        return []
    pairs = [(query, s) for s in sents]
    scores = _cross.predict(pairs).tolist()
    scored = list(zip(sents, [float(x) for x in scores]))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [(sent, score) for sent, score in scored[:k] if score >= min_score]

def _compress_doc(query: str, doc: dict, mode: str, per_doc: int, min_score: float) -> list[str]:
    """
    doc = {"title","url","snippet","score"}
    returns list of selected sentences (strings) for this doc.
    """
    snippet = doc.get("snippet") or ""
    if not snippet:
        return []
    if mode == "crossenc":
        picks = _top_sentences_crossenc(query, snippet, per_doc * 2, min_score)
    else:
        picks = _top_sentences_embed(query, snippet, per_doc * 3, min_score)
    # keep at most per_doc, preserve original order for readability
    sent_set = set(s for s, _ in picks[: per_doc * 4])
    ordered = [s for s in _split_sentences(snippet) if s in sent_set]
    return ordered[:per_doc]

def _build_compressed_context(query: str, docs: list[dict], mode: str, per_doc: int,
                              min_score: float, max_chars: int) -> tuple[str, list[dict]]:
    """
    Build a numbered context with selected sentences per doc.
    Returns (context_text, sources_list).
    """
    lines = [f"Question: {query}", "", "Context (numbered snippets):"]
    sources = []
    total = 0
    idx = 1
    for d in docs:
        url = d.get("url") or ""
        title = d.get("title") or ""
        chosen = _compress_doc(query, d, mode=mode, per_doc=per_doc, min_score=min_score)
        if not chosen:
            continue
        block = f"[{idx}] {title}\nURL: {url}\n" + " ".join(chosen) + "\n"
        add_len = len(block)
        if total + add_len > max_chars and total > 0:
            break
        lines.append(block)
        sources.append({"title": title, "url": url, "score": d.get("score")})
        total += add_len
        idx += 1
    return ("\n".join(lines), sources)
