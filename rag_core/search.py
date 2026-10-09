"""
公共检索模块：服务端和评估脚本共用。

- LocalEmbeddings: 封装 bge-small 本地 embedding
- load_vectorstore: 加载 FAISS 索引
- build_bm25: 构建 BM25 索引
- hybrid_search: 混合检索（向量 + BM25 + 加权 RRF），含距离阈值降级
- diverse_by_source: MMR + 按 source 去重
"""
import os
import sys
from pathlib import Path

# ===== HF 环境初始化（必须在 sentence_transformers 之前）=====
def _resolve_hf_home() -> str:
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    container_default = Path("/root/.cache/huggingface")
    if container_default.exists():
        return str(container_default)
    return str(Path(__file__).resolve().parent.parent / "hf_cache")

_HF_HOME = _resolve_hf_home()
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


# 向量 L2 距离阈值，超过认为不相关（bge 归一化后范围 0~2）
MAX_DISTANCE = 1.1

# 默认模型和索引位置
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # 让 HF 自己根据 HF_HOME 解析缓存路径
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def load_vectorstore(index_dir=None, embedding=None):
    """加载 FAISS 索引。index_dir 和 embedding 都可选（评估脚本可复用）。"""
    if index_dir is None:
        index_dir = DEFAULT_INDEX_DIR
    if embedding is None:
        embedding = LocalEmbeddings()
    return FAISS.load_local(
        str(index_dir),
        embedding,
        allow_dangerous_deserialization=True,
    )


def build_bm25(vectorstore):
    """从 vectorstore 里取出所有 chunk，构建 BM25 索引。"""
    all_docs = list(vectorstore.docstore._dict.values())
    tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_docs


def hybrid_search(
    query,
    vectorstore,
    bm25,
    all_docs,
    k: int = 3,
    rrf_k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    max_distance: float = MAX_DISTANCE,
    verbose: bool = False,
):
    """
    混合检索：向量 + BM25 + 加权 RRF，含距离阈值降级。

    verbose=True 时打印最佳向量距离和降级信息到 stderr（服务端用）。
    """
    # 向量检索（带距离分数）
    vector_results_with_scores = vectorstore.similarity_search_with_score(query, k=k * 3)

    if not vector_results_with_scores:
        return []

    # 降级判断：最佳向量距离太大 → 认为知识库中无相关内容
    best_distance = vector_results_with_scores[0][1]
    if verbose:
        print(f"[RAG] 最佳向量距离: {best_distance:.4f}", file=sys.stderr)

    if best_distance > max_distance:
        if verbose:
            print(f"[RAG] 降级触发（距离 {best_distance:.4f} > {max_distance}）", file=sys.stderr)
        return []

    # 拆出文档
    vector_results = [doc for doc, _ in vector_results_with_scores]
    vector_ranks = {doc.page_content: i for i, doc in enumerate(vector_results)}

    # BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[: k * 3]
    bm25_ranks = {all_docs[i].page_content: rank for rank, i in enumerate(bm25_top_indices)}

    # 加权 RRF 合并
    all_contents = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores = {}
    for content in all_contents:
        score = 0.0
        if content in vector_ranks:
            score += vector_weight / (rrf_k + vector_ranks[content])
        if content in bm25_ranks:
            score += bm25_weight / (rrf_k + bm25_ranks[content])
        rrf_scores[content] = score

    top_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:k]
    content_to_doc = {doc.page_content: doc for doc in all_docs}
    return [content_to_doc[c] for c in top_contents]


def diverse_by_source(query, vectorstore, k: int = 3):
    """MMR + 按 source 去重（评估脚本用）。"""
    results = vectorstore.max_marginal_relevance_search(query, k=k * 3, fetch_k=10)
    seen = set()
    out = []
    for doc in results:
        src = doc.metadata["source"]
        if src not in seen:
            out.append(doc)
            seen.add(src)
            if len(out) == k:
                break
    return out