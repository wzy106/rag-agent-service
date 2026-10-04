import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import jieba
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ===== 加载 embedding 和 FAISS =====
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

# ===== 构建 BM25 索引 =====
all_docs = list(vectorstore.docstore._dict.values())
print(f"从 FAISS 加载了 {len(all_docs)} 个 chunk")

tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)

# ===== 混合检索 =====
def hybrid_search(query, k=3, rrf_k=60, vector_weight=0.7, bm25_weight=0.3):
    vector_results = vectorstore.similarity_search(query, k=k*3)
    vector_ranks = {doc.page_content: i for i, doc in enumerate(vector_results)}

    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*3]
    bm25_ranks = {all_docs[i].page_content: rank for rank, i in enumerate(bm25_top_indices)}

    all_contents = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores = {}
    for content in all_contents:
        score = 0
        if content in vector_ranks:
            score += vector_weight / (rrf_k + vector_ranks[content])
        if content in bm25_ranks:
            score += bm25_weight / (rrf_k + bm25_ranks[content])
        rrf_scores[content] = score

    top_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:k]
    content_to_doc = {doc.page_content: doc for doc in all_docs}
    return [content_to_doc[c] for c in top_contents]

# ===== 评估对比 =====
with open(Path(__file__).resolve().parent / "test_set.json", encoding="utf-8") as f:
    test_set = json.load(f)

def evaluate(search_fn, name, k=3):
    hit = 0
    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]
        if any(exp in sources for exp in item["expected_sources"]):
            hit += 1
        else:
            print(f"  ❌ {item['question']} → 期望 {item['expected_sources']}，实际 {sources}")
    recall = hit / len(test_set)
    print(f"\n[{name}] 召回率：{hit}/{len(test_set)} = {recall:.0%}")

print("=== 普通相似度 ===")
evaluate(lambda q, k: vectorstore.similarity_search(q, k=k), "相似度")

print("\n=== MMR ===")
evaluate(lambda q, k: vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=10), "MMR")

print("\n=== 混合检索（向量 + BM25 + RRF）===")
evaluate(hybrid_search, "混合检索")