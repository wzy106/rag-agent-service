import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
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

with open(Path(__file__).resolve().parent / "test_set.json", encoding="utf-8") as f:
    test_set = json.load(f)

# ===== 基础评估：召回率 =====
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
    return recall

print("=== 评估：普通相似度检索 ===")
evaluate(lambda q, k: vectorstore.similarity_search(q, k=k), "相似度")

print("\n=== 评估：MMR 检索 ===")
evaluate(lambda q, k: vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=10), "MMR")

# ===== 多样性对比 =====
def diverse_search(query, k=3):
    results = vectorstore.max_marginal_relevance_search(query, k=k*3, fetch_k=10)
    seen_sources = set()
    diverse_results = []
    for doc in results:
        src = doc.metadata["source"]
        if src not in seen_sources:
            diverse_results.append(doc)
            seen_sources.add(src)
            if len(diverse_results) == k:
                break
    return diverse_results

def measure_diversity(search_fn, name, k=3):
    total = 0
    for item in test_set:
        results = search_fn(item["question"], k)
        sources = set(doc.metadata["source"] for doc in results)
        total += len(sources)
    avg = total / len(test_set)
    print(f"[{name}] 平均多样性：{avg:.2f} 个不同文档/查询")
    return avg

print("\n=== 多样性对比（Top-3 里平均有几个不同文档）===")
measure_diversity(lambda q, k: vectorstore.similarity_search(q, k=k), "相似度")
measure_diversity(lambda q, k: vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=10), "MMR")
measure_diversity(diverse_search, "MMR + 按来源去重")

# ===== 完整评估：召回率 + 精确率 + MRR =====
def evaluate_full(search_fn, name, k=3):
    hit = 0
    precision_sum = 0
    mrr_sum = 0

    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]

        # 召回率：期望文档有没有出现
        if any(exp in sources for exp in item["expected_sources"]):
            hit += 1

        # 精确率：Top-k 里有多少是期望文档
        if sources:
            relevant = sum(1 for s in sources if s in item["expected_sources"])
            precision_sum += relevant / len(sources)

        # MRR：第一个正确结果的排名倒数
        for rank, s in enumerate(sources, 1):
            if s in item["expected_sources"]:
                mrr_sum += 1 / rank
                break

    n = len(test_set)
    print(f"\n[{name}]")
    print(f"  召回率：{hit}/{n} = {hit/n:.0%}")
    print(f"  精确率：{precision_sum/n:.2f}")
    print(f"  MRR：{mrr_sum/n:.2f}")
    return hit/n, precision_sum/n, mrr_sum/n

print("\n\n=== 三种指标的完整评估 ===")
evaluate_full(lambda q, k: vectorstore.similarity_search(q, k=k), "相似度")
evaluate_full(lambda q, k: vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=10), "MMR")
evaluate_full(diverse_search, "MMR + 按来源去重")