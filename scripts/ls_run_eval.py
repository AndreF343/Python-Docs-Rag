import os
import requests
from langsmith import Client, evaluation as ls_eval

API = os.getenv("API_BASE", "http://127.0.0.1:8000")
DATASET = os.getenv("DATASET", "py-docs-retrieval-smoke")
API_KEY = os.getenv("API_KEY", "")  # FastAPI key if enabled

def run_example(inputs: dict, *, langsmith_extra=None) -> dict:
    """Target under test: LangSmith passes INPUTS as a dict."""
    q = inputs["query"]
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    r = requests.post(
        f"{API}/search",
        json={"query": q, "k_retrieve": 20, "k_final": 5},
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()  # {"results":[{"url":...}, ...]}

def _norm(u: str) -> str:
    if not u:
        return ""
    u = u.strip().split("#", 1)[0].split("?", 1)[0]
    u = u.replace("http://", "https://")
    return u.rstrip("/")

def judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs) -> list[dict]:
    """
    Return a list of evaluation results, one per metric:
      {"key": <metric_name>, "score": <float>}  # numeric
      {"key": <metric_name>, "value": <str>}    # categorical/text
    """
    def _norm(u: str) -> str:
        if not u:
            return ""
        u = u.strip().split("#", 1)[0].split("?", 1)[0]
        u = u.replace("http://", "https://")
        return u.rstrip("/")

    truth = _norm((reference_outputs or {}).get("doc_url", ""))
    urls = [_norm(x.get("url", "")) for x in (outputs or {}).get("results", [])]
    hit5 = float(truth in urls[:5]) if truth else 0.0
    top1 = urls[0] if urls else ""

    return [
        {"key": "retrieval_hit@5", "score": hit5},
        {"key": "top1_url", "value": top1},
    ]


if __name__ == "__main__":
    client = Client()
    dataset = client.read_dataset(dataset_name=DATASET)

    ls_eval.evaluate(
        run_example,                 # target
        data=dataset,                # Dataset object (provides inputs & reference_outputs)
        evaluators=[judge],          # custom evaluator (LS-compatible args)
        experiment_prefix="py-docs-rag-eval",
    )
    print("Eval run submitted.")
