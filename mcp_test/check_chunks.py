import os
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class LocalEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

model_path = str(next((Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots").iterdir()))
embedding = LocalEmbeddings(model_path)

INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
vectorstore = FAISS.load_local(str(INDEX_DIR), embedding, allow_dangerous_deserialization=True)

counter = Counter(d.metadata["source"] for d in vectorstore.docstore._dict.values())
print("Chunk 分布：")
for source, count in counter.most_common():
    print(f"  {source}: {count} 块")