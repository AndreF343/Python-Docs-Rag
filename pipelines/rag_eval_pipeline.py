from kfp import dsl
from kfp.dsl import pipeline
# ✅ v2 extension import
from kfp import kubernetes

IMAGE = "py-docs-pipeline:local"  # your local pipeline image with repo copied in

@dsl.container_component
def build_index_comp():
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "scripts/build_index.py"],
        args=[],
    )

@dsl.container_component
def eval_comp():
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "scripts/eval_retrieval_rerank_raw_mlflow.py"],
        args=[],
    )

@pipeline(name="py-docs-build-and-eval")
def rag_build_and_eval():
    b = build_index_comp()
    e = eval_comp().after(b)

    # ✅ Mount PVCs onto tasks (v2 way via kfp-kubernetes)
    kubernetes.mount_pvc(task=b, pvc_name="models-pvc", mount_path="/models")

    kubernetes.mount_pvc(task=e, pvc_name="models-pvc", mount_path="/models")
    kubernetes.mount_pvc(task=e, pvc_name="mlruns-pvc", mount_path="/mlruns")

    # ✅ Task-level env (v2 supports this)
    e.set_env_variable(name="MLFLOW_TRACKING_URI", value="file:///mlruns")
    e.set_env_variable(name="HF_HOME", value="/models")
    e.set_env_variable(name="SENTENCE_TRANSFORMERS_HOME", value="/models")
    e.set_env_variable(name="TRANSFORMERS_CACHE", value="/models")

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=rag_build_and_eval,
        package_path="pipelines/rag_build_and_eval.yaml"
    )

@dsl.container_component
def build_bm25_comp():
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["/bin/sh","-lc"],
        args=["set -ex; python scripts/build_bm25.py"],
        pod_spec_patch=POD_PATCH,
    )

@pipeline(name="py-docs-build-and-eval")
def rag_build_and_eval():
    b1 = build_index_comp()
    b2 = build_bm25_comp().after(b1)
    e  = eval_comp().after(b2)
