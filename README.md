# Python Docs RAG — Retrieval + Re-ranking + Citations (K8s, MLflow, KFP, LangSmith)

**What**  
Containerized, end-to-end **RAG** over the official Python docs:
- **Vector retrieval** (Sentence-Transformers) + **BM25** (hybrid via **RRF**)
- **Cross-encoder re-ranking**
- **`/answer`** endpoint: compressed, query-aware context → **local LLM (Ollama)** → **citations**
- Evaluation with **LangSmith** & **MLflow**, automated in **Kubeflow Pipelines**
- Deployed locally and on **Kubernetes (Minikube)**

**Why**  
Demonstrates practical IR quality improvements (re-ranking), experiment tracking, and Kubernetes deployment in a compact project.

---

## Notable Features and Tools

- Quality (example): `Recall@5 ≈ 0.xx–0.xx`, `MRR@5 ≈ 0.xx–0.xx` on a 300-query set (update with your final numbers).
- **Hybrid retrieval:** BM25 + vectors with **Reciprocal Rank Fusion** improves exact-keyword/symbol queries.
- **Answer synthesis w/ citations:** Local **Ollama** model (e.g., `qwen2.5:7b`) on **compressed** context for speed and faithfulness.
- **Observability:** LangSmith traces/evals + MLflow metrics/runs + KFP pipeline (build → eval).
- **Kubernetes:** FastAPI on Minikube with PVC-backed model cache and hostPath index mount.
- **Automation:** KFP pipeline: build index → run eval; MLflow logs metrics for each run.
  
---

## Architecture

Ingestion → Chroma (HNSW) → Retrieval (SBERT) + BM25 → **RRF fusion** → Cross-Encoder Re-rank → `/search` & `/answer` (LLM + citations)

Ingestion → Chroma (HNSW) → Retrieval (SBERT) → Re-rank (CrossEncoder) → FastAPI  
Ops: MLflow (metrics + Registry), LangSmith (traces + eval), KFP (build→eval pipeline), Minikube (Deployment/Service)


> _Add an architecture diagram screenshot here (PNG)._

- Control plane: **KFP** to run build/eval, **MLflow** to log metrics, **LangSmith** to trace runs/evals.
- Data plane: Chroma (persisted), Sentence-Transformers for embeddings & cross-encoder; BM25 for lexical signals.

---

## Repo Layout (key files)
```
app/
  main.py                 # FastAPI service (/search, /answer, /health)
  settings.py             # Env toggles (HYBRID, COMPRESS, models, caches)
scripts/
  build_index_raw.py      # Build Chroma index from JSONL chunks
  build_bm25.py           # Build BM25 index (rank_bm25)
  demo_cli.py             # CLI demo hitting /search
  streamlit_app.py        # One-page UI hitting the API
  ls_make_dataset.py      # LangSmith dataset create
  ls_run_eval.py          # LangSmith evaluation run
  eval_retrieval_rerank_raw_mlflow.py  # MLflow eval (Recall/MRR)
data/
  chunks/python_docs_chunks.jsonl
  index/chroma/           # Chroma persistent store
  index/bm25/             # BM25 pickles
k8s/
  deployment.yaml         # API Deployment/Service for Minikube
  pvc-models.yaml         # PVC for model cache
```
---

## Quickstart (local, Windows CMD)

```
REM 1) Create venv & install deps
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

REM 2) Build indices (Chroma + BM25)
python scripts\build_index_raw.py
python scripts\build_bm25.py

REM 3) Pull a local LLM once (for /answer)
ollama pull qwen2.5:7b

REM 4) Run API (hybrid + compression ON by default here)
set HYBRID=true
set COMPRESS=true
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## Demo CLI

python scripts\demo_cli.py "How do I create a virtual environment?"

## Smple Web UI

```
pip install streamlit requests
streamlit run scripts\streamlit_app.py
```
Open http://127.0.0.1:8501

## Endpoints

- `POST` /search → retrieves, fuses, reranks → returns top-k with scores & snippets.

- `POST` /answer → same, plus context compression → Ollama → short answer with [n] citations and full URLs.

- `GET` /health → status, model names, index location.

Examples (CMD)

```
curl -s -X POST http://127.0.0.1:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"typing.NamedTuple\",\"k_retrieve\":20,\"k_final\":5}" | more
```
```
curl -s -X POST http://127.0.0.1:8000/answer ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"How do I create a virtual environment?\",\"k_retrieve\":20,\"k_final\":5}" | more
```
---

## Results



## Roadmap
- Split reranker via KServe for canary & autoscale.
- Swap vector store (pgvector/Qdrant) for remote persistence.
- Tiny fine-tune of reranker; register as Model v2 in MLflow.
- API key auth (toggle via API_KEY env) — simple header check already scaffolded.
- Change data source to something more interesting and complex like the Warhammer 40K universe.

## Troublshooting
- `ModuleNotFoundError: langchain` in KFP build step → Use `build_index_raw.py` (no LangChain), or add `langchain` to the KFP image.
- “attempt to write a readonly database” (Chroma) → mount with `--uid=0 --gid=0`; verify write perms to `/workspace`.
- KFP pod Pending: PVC not found → create `models-pvc` in `kubeflow` namespace or remove PVC from the step.
- Ollama in K8s → point `OLLAMA_BASE` to `http://host.minikube.internal:11434` or run Ollama as a Pod/DaemonSet.

## Acknowledgements
- Sentence-Transformers: <a href="https://www.sbert.net">https://www.sbert.net
</a>

- Chroma: <a href="https://www.trychroma.com">https://www.trychroma.com
</a>

- LangSmith: <a href="https://smith.langchain.com">https://smith.langchain.com
</a>

- MLflow: <a href="https://mlflow.org">https://mlflow.org
</a>

- Kubeflow Pipelines: <a href="https://www.kubeflow.org/docs/components/pipelines/">https://www.kubeflow.org/docs/components/pipelines/
</a>

- Ollama: <a href="https://ollama.com">https://ollama.com
</a>

