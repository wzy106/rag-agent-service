"""
RAG 检索策略评估脚本

跑法：
    cd D:/Projects/Agent
    python eval/eval_retrieval.py

输出：
- 随机基线（组合公式精确计算）
- 4 种策略的 4 指标对比（召回率 / 精确率 / MRR / 多样性）
- 汇总表

注：所有检索函数从 rag_core.search 导入，保证评估跑的就是服务端部署的代码。
"""
import os
import sys
import json
from math import comb
from collections import Counter
from pathlib import Path

# ===== 确保能找到项目根的 rag_core 包 =====
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import (
    load_vectorstore,
    build_bm25,
    hybrid_search,
    diverse_by_source,
    LocalEmbeddings,
    DEFAULT_MODEL,
)


# ===== 加载向量库和 BM25 索引 =====
print("加载 embedding 和 FAISS 索引...")
embedding = LocalEmbeddings(DEFAULT_MODEL)
vectorstore = load_vectorstore(embedding=embedding)
bm25, all_docs = build_bm25(vectorstore)
print(f"加载 {len(all_docs)} 个 chunk\n")

with open(Path(__file__).resolve().parent / "test_set.json", encoding="utf-8") as f:
    test_set = json.load(f)
print(f"加载 {len(test_set)} 个测试项\n")


# ===== 随机基线 =====
def random_baseline(test_set, all_docs, k=3):
    """
    从全部 chunk 里随机抽 k 个，算 source 级召回率期望。

    用组合公式精确计算，不用模拟：
        P(至少命中一个期望 source) = 1 - C(N - n_expected, k) / C(N, k)

    其中 N = 总 chunk 数，n_expected = 期望 source 的 chunk 数。
    """
    n_total = len(all_docs)
    source_counts = Counter(doc.metadata["source"] for doc in all_docs)

    total = 0.0
    for item in test_set:
        n_expected = sum(source_counts.get(s, 0) for s in item["expected_sources"])
        if n_total - n_expected < k:
            prob = 1.0
        else:
            prob = 1 - comb(n_total - n_expected, k) / comb(n_total, k)
        total += prob
    return total / len(test_set)


# ===== 四种检索策略（都用公共实现）=====
def search_similarity(q, k):
    return vectorstore.similarity_search(q, k=k)

def search_mmr(q, k):
    return vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=10)

def search_diverse(q, k):
    return diverse_by_source(q, vectorstore, k=k)

def search_hybrid(q, k):
    return hybrid_search(q, vectorstore, bm25, all_docs, k=k, verbose=False)


# ===== 四指标评估 =====
def evaluate_full(search_fn, name, k=3):
    hit = 0
    precision_sum = 0.0
    mrr_sum = 0.0
    diversity_sum = 0

    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]
        unique_sources = set(sources)

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

        # 多样性：Top-k 里不同 source 数
        diversity_sum += len(unique_sources)

    n = len(test_set)
    return {
        "recall": hit / n,
        "precision": precision_sum / n,
        "mrr": mrr_sum / n,
        "diversity": diversity_sum / n,
    }


# ===== 跑评估 =====
print("=" * 70)
print("RAG 检索策略评估")
print("=" * 70)

baseline = random_baseline(test_set, all_docs, k=3)
print(f"\n[随机基线] 召回率：{baseline:.1%}")
print("  （从 102 个 chunk 里随机抽 3 个，命中期望 source 的概率）\n")

strategies = [
    ("普通相似度", search_similarity),
    ("MMR", search_mmr),
    ("MMR + 按来源去重", search_diverse),
    ("混合检索（向量+BM25+RRF）", search_hybrid),
]

results = []
for name, fn in strategies:
    r = evaluate_full(fn, name, k=3)
    results.append((name, r))
    print(f"[{name}]")
    print(f"  召回率：{r['recall']:.1%}")
    print(f"  精确率：{r['precision']:.2f}")
    print(f"  MRR：   {r['mrr']:.2f}")
    print(f"  多样性：{r['diversity']:.2f}")
    print()

# ===== 汇总表 =====
print("=" * 70)
print("汇总表")
print("=" * 70)
print(f"{'策略':<30} {'召回率':>8} {'精确率':>8} {'MRR':>8} {'多样性':>8}")
print("-" * 70)
for name, r in results:
    print(
        f"{name:<30} "
        f"{r['recall']:>7.1%} "
        f"{r['precision']:>8.2f} "
        f"{r['mrr']:>8.2f} "
        f"{r['diversity']:>8.2f}"
    )
print("-" * 70)
print(f"{'随机基线':<30} {baseline:>7.1%}")