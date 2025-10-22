import os, json, tempfile, mlflow
from pathlib import Path
from typing import Dict, Any

# --- Current config (pull from env with sensible defaults) ---
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index" / "chroma"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "py_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
K_RETRIEVE = int(os.getenv("K_RETRIEVE", "20"))
K_FINAL = int(os.getenv("K_FINAL", "5"))

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "py-docs-rag")

def current_config() -> Dict[str, Any]:
    return {
        "index_dir": str(INDEX_DIR),
        "collection": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "cross_encoder": CROSS_ENCODER_MODEL,
        "k_retrieve": K_RETRIEVE,
        "k_final": K_FINAL,
    }

# --- a tiny pyfunc that just returns the config (placeholder) ---
import mlflow.pyfunc

class RAGConfigModel(mlflow.pyfunc.PythonModel):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
    def predict(self, context, model_input=None):
        return self.cfg

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("py-docs-rag-registry")

    cfg = current_config()

    with mlflow.start_run(run_name="register_rag_config") as run:
        # Log params and a config artifact
        mlflow.log_params({
            "embed_model": cfg["embed_model"],
            "cross_encoder": cfg["cross_encoder"],
            "k_retrieve": cfg["k_retrieve"],
            "k_final": cfg["k_final"],
            "collection": cfg["collection"],
        })
        tmpd = tempfile.mkdtemp()
        cfg_path = Path(tmpd) / "rag_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        # Log a minimal pyfunc "model" containing this config
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RAGConfigModel(cfg),
            pip_requirements=["mlflow", "cloudpickle"],
            registered_model_name=None,  # register explicitly below
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        print("Logged model_uri:", model_uri)

        # Register in the Model Registry
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print("Registered model:", mv.name, "version:", mv.version)

        # Optionally transition to STAGING
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME, version=mv.version, stage="Staging", archive_existing_versions=False
        )
        print("Transitioned to STAGING")

if __name__ == "__main__":
    main()
