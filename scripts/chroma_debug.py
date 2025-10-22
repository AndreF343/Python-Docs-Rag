# scripts/chroma_debug.py
import os
from chromadb import PersistentClient

INDEX_DIR = os.getenv("INDEX_DIR") or r".\data\index\chroma"
INDEX_DIR = os.path.abspath(INDEX_DIR)
print("INDEX_DIR =", INDEX_DIR)

client = PersistentClient(path=INDEX_DIR)
cols = client.list_collections()
if not cols:
    print("No collections found.")
else:
    for c in cols:
        try:
            print(f"- name={c.name} id={c.id} count={c.count()}")
        except Exception as e:
            print(f"- name={c.name} id={c.id} (count error: {e})")
