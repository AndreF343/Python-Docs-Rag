import argparse, pathlib
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from chromadb.config import Settings
from langchain_chroma import Chroma


INDEX_DIR = pathlib.Path("data/index/chroma")

class SBertEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="natural language query")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    vs = Chroma(collection_name="py_docs",
                embedding_function=SBertEmbeddings(),
                persist_directory=str(INDEX_DIR),
                client_settings=Settings(anonymized_telemetry=False))
    
    print("INDEX_DIR:", INDEX_DIR)
    print("Querying…")
    docs = vs.similarity_search(args.query, k=args.k)
    print("Got", len(docs), "docs")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        print(f"\n[{i}] {meta.get('title','(no title)')}")
        print(f"URL: {meta.get('doc_url')}")
        print(d.page_content[:400].replace("\n"," ") + ("..." if len(d.page_content)>400 else ""))

if __name__ == "__main__":
    main()
