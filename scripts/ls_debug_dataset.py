from langsmith import Client
import os

DATASET = os.getenv("DATASET", "py-docs-retrieval-smoke")
client = Client()

# Find dataset by name in the current project
ds = client.read_dataset(dataset_name=DATASET)
print("Dataset:", ds.name, "id:", ds.id)

# Pull examples and count doc_url presence
examples = list(client.list_examples(dataset_id=ds.id))
print("Example count:", len(examples))

missing, sample = 0, None
for ex in examples:
    outputs = ex.outputs or {}
    if "doc_url" not in outputs or not (outputs.get("doc_url") or "").strip():
        missing += 1
        if not sample:
            sample = ex
print("Missing doc_url:", missing)
if sample:
    print("First missing example:")
    print(" inputs:", sample.inputs)
    print(" outputs:", sample.outputs)
else:
    print("All examples have doc_url ✅")
