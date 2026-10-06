import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
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

# ===== 测各种问题的距离 =====
test_queries = [
    "MMR 是什么？",                          # 正常
    "RAG 的完整流程是什么？",                 # 正常
    "Docker 怎么部署？",                      # 正常
    "今天天气怎么样？",                       # 无关
    "如何做红烧肉？",                         # 无关
    "明天股市会涨吗？",                       # 无关
    "推荐几部好电影",                         # 无关
    "asdfghjkl",                             # 乱码
]

print(f"{'问题':<30} {'最佳距离':<12} {'是否触发降级(>1.2)'}")
print("=" * 70)

for query in test_queries:
    results = vectorstore.similarity_search_with_score(query, k=1)
    if results:
        distance = results[0][1]
        fallback = "✅ 触发" if distance > 1.2 else "❌ 不触发"
        print(f"{query:<30} {distance:<12.4f} {fallback}")
    else:
        print(f"{query:<30} 无结果")