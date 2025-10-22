from pathlib import Path
import os, pathlib

# Defaults point to your repo; override via env if needed
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "data" / "index" / "chroma"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "py_docs")

# Models
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval knobs
K_RETRIEVE_DEFAULT = int(os.environ.get("K_RETRIEVE", "20"))
K_FINAL_DEFAULT = int(os.environ.get("K_FINAL", "5"))


MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "")  # e.g., "models:/py-docs-rag/Staging"

# API_KEY = os.getenv("API_KEY", "")         # if set, require it
# ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")  # comma-separated

# --- LLM answer settings (Ollama) ---
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # or llama3.2:3b
MAX_TOKENS_ANSWER = int(os.getenv("MAX_TOKENS_ANSWER", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# threshold to decide if we should answer at all
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.15"))

HYBRID = os.getenv("HYBRID", "false").lower() in ("1","true","yes","y")
# how many candidates from each retriever before fusion
VEC_CAND = int(os.getenv("VEC_CAND", "50"))
BM25_CAND = int(os.getenv("BM25_CAND", "50"))
# RRF constant k (larger = less steep penalty)
RRF_K = int(os.getenv("RRF_K", "60"))

# BM25 paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
BM25_DIR = pathlib.Path(os.getenv("BM25_DIR", ROOT / "data" / "index" / "bm25"))

# --- Context compression (for /answer only) ---
COMPRESS = os.getenv("COMPRESS", "true").lower() in ("1","true","yes","y")
# "embed" (fast cosine using the SentenceTransformer) or "crossenc" (slower but sharper)
COMPRESS_MODE = os.getenv("COMPRESS_MODE", "embed")  # embed|crossenc
SENTS_PER_DOC = int(os.getenv("SENTS_PER_DOC", "3"))  # how many sentences to keep per doc
MIN_SENT_SCORE = float(os.getenv("MIN_SENT_SCORE", "0.25"))  # filter weak sentences
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))  # hard cap for prompt context
