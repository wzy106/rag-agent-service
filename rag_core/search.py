"""
公共检索模块：服务端和评估脚本共用。

- LocalEmbeddings: 封装 bge-small 本地 embedding
- load_vectorstore: 加载 FAISS 索引
- build_bm25: 构建 BM25 索引
- hybrid_search: 混合检索（向量 + BM25 + 加权 RRF），含距离阈值降级
- diverse_by_source: MMR + 按 source 去重
- rerank: 用 bge-reranker-base 精排（CrossEncoder）
- hybrid_search_with_rerank: 混合检索召回 + Rerank 精排的完整 pipeline
"""
import os
import sys
from pathlib import Path

# ===== HF 环境初始化（必须在 sentence_transformers 之前）=====
def _safe_exists(path: Path) -> bool:
    """Path.exists() can raise PermissionError for /root on CI runners."""
    try:
        return path.exists()
    except OSError:
        return False


def _resolve_hf_home() -> str:
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    container_default = Path("/root/.cache/huggingface")
    if _safe_exists(container_default):
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

from rag_core.chunking import derive_chunk_id, normalize_content


# 向量 L2 距离阈值，超过认为不相关（bge 归一化后范围 0~2）
# 通过 eval/calibrate_distance.py 标定（2026-10-12，新语料 576 chunk）：
# 正样本 100 条 + 负样本 25 条
# 阈值 0.96 是 F1 最大点：P=0.990, R=1.000, F1=0.995
# 相比 1.0（P=0.980, R=1.000），精确率 +1%
MAX_DISTANCE = 0.96

# 默认模型和索引位置
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

# Rerank 模型
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ===== 检索 / 精排配置（环境变量可覆盖）=====
# 服务端和评估脚本共用同一份配置，保证"评估跑的就是部署的代码"。

# 粗排召回候选数：Rerank 耗时与候选数近似线性，候选越少越快，但会伤召回率。
# 实测（249 条测试集）：k=20 时 Rerank 召回 57.0%；k=10 时掉到 53.8%，且 Rerank 增益
# 从 +4.4 塌缩到 +1.2。故默认保留 20；追求极致延迟可调小（代价是召回）。
RECALL_K = int(os.getenv("RAG_RECALL_K", "20"))
# Reranker 输入截断长度（token），越大越准也越慢。
RERANK_MAX_LENGTH = int(os.getenv("RAG_RERANK_MAX_LENGTH", "512"))
# 精排开关：置 0/false/no 则跳过 Rerank，直接返回粗排 Top-k（快速降级用）。
ENABLE_RERANK = os.getenv("RAG_ENABLE_RERANK", "1").lower() not in ("0", "false", "no")


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # 懒加载：让 rag_core.search 在没装 torch 的环境里也能被导入（单测/CI）。
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers 需要 torch；请先安装 CPU 版 torch"
            ) from exc
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


def doc_key(doc) -> str:
    """Stable identity for RRF maps. Never use page_content as the key."""
    return derive_chunk_id(doc)


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
    dedupe_content: bool = True,
):
    """
    混合检索：向量 + BM25 + 加权 RRF，含距离阈值降级。

    verbose=True 时打印最佳向量距离和降级信息到 stderr（服务端用）。
    """
    # 向量检索（带距离分数）
    vector_results_with_scores = vectorstore.similarity_search_with_score(query, k=k * 3)

    if not vector_results_with_scores:
        return []

    # 降级判断：最佳向量距离太大 → 认为知识库中无相关内容。
    # 已知限制：这是 top-1 单点判断，且发生在 BM25/RRF 之前；关键词强相关
    # 但向量距离大的 query 会被提前拒绝。本轮保持行为不变，仅在评估里量化。
    best_distance = vector_results_with_scores[0][1]
    if verbose:
        print(f"[RAG] 最佳向量距离: {best_distance:.4f}", file=sys.stderr)

    if best_distance > max_distance:
        if verbose:
            print(f"[RAG] 降级触发（距离 {best_distance:.4f} > {max_distance}）", file=sys.stderr)
        return []

    # 拆出文档
    vector_results = [doc for doc, _ in vector_results_with_scores]
    vector_ranks = {doc_key(doc): i for i, doc in enumerate(vector_results)}

    # BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[: k * 3]
    bm25_ranks = {doc_key(all_docs[i]): rank for rank, i in enumerate(bm25_top_indices)}

    # 加权 RRF 合并（用稳定 chunk_id 做 key，避免相同内容跨文档互相覆盖）
    all_keys = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores = {}
    for key in all_keys:
        score = 0.0
        if key in vector_ranks:
            score += vector_weight / (rrf_k + vector_ranks[key])
        if key in bm25_ranks:
            score += bm25_weight / (rrf_k + bm25_ranks[key])
        rrf_scores[key] = score

    ranked_keys = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)
    key_to_doc = {doc_key(doc): doc for doc in all_docs}
    for doc in vector_results:
        key_to_doc.setdefault(doc_key(doc), doc)

    if not dedupe_content:
        return [key_to_doc[key] for key in ranked_keys[:k] if key in key_to_doc]

    # 相同内容只保留 RRF 排名最高的副本，继续向后取满 k 条。
    out = []
    seen = set()
    for key in ranked_keys:
        doc = key_to_doc.get(key)
        if doc is None:
            continue
        fingerprint = normalize_content(doc.page_content)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(doc)
        if len(out) == k:
            break
    return out


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


# ============================================================
# Rerank（精排）
# ============================================================

# 懒加载：第一次调用 rerank 时才加载 3 GB 模型，避免 import 时拖慢启动
_reranker = None


def _get_reranker():
    """懒加载 CrossEncoder。第一次调用时加载，之后复用。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] 加载 {RERANKER_MODEL}（max_length={RERANK_MAX_LENGTH}）...", file=sys.stderr)
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=RERANK_MAX_LENGTH)
        print(f"[Reranker] 加载完成", file=sys.stderr)
    return _reranker


def rerank(query: str, docs: list, top_k: int = 3, verbose: bool = False) -> list:
    """
    用 bge-reranker-base 对候选 docs 精排，返回 top_k。

    bge-reranker 是 CrossEncoder（交叉编码器），输入 (query, doc) 对，
    输出相关性分数。比双塔 embedding 更准，但更慢，所以只用于精排 Top-K。

    参数：
        query: 用户问题
        docs: 混合检索召回的候选文档列表（建议 20 条左右）
        top_k: 精排后保留几条
        verbose: 是否打印分数到 stderr（服务端调试用）

    返回：
        按 reranker 分数从高到低排序的 top_k 个文档
    """
    if not docs:
        return []

    reranker = _get_reranker()
    # CrossEncoder 输入格式：[(query, doc_content), ...]
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    # 按分数降序排序，取 top_k
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)

    if verbose:
        print(f"[Rerank] 候选 {len(docs)} 条 → Top-{top_k}", file=sys.stderr)
        for i, (doc, score) in enumerate(ranked[:top_k], 1):
            src = doc.metadata.get("source", "?")
            print(f"  [{i}] score={float(score):.4f} source={src}", file=sys.stderr)

    return [doc for doc, _ in ranked[:top_k]]


def hybrid_search_with_rerank(
    query,
    vectorstore,
    bm25,
    all_docs,
    recall_k: int | None = None,
    final_k: int = 3,
    verbose: bool = False,
    **hybrid_kwargs,
) -> list:
    """
    混合检索 + Rerank 的完整 pipeline（两阶段检索）。

    流程：
        1. hybrid_search 粗排召回 recall_k 条（默认取配置 RECALL_K）
        2. rerank 精排到 final_k 条（默认 3）；ENABLE_RERANK=0 时跳过本步

    参数：
        recall_k: 粗排阶段保留多少条；None 时用配置 RECALL_K
        final_k: 精排后返回多少条（默认 3）
        verbose: 是否打印中间过程到 stderr
        **hybrid_kwargs: 传给 hybrid_search 的其他参数（rrf_k、权重等）
    """
    if recall_k is None:
        recall_k = RECALL_K

    # 1. 粗排：混合检索召回
    candidates = hybrid_search(
        query,
        vectorstore=vectorstore,
        bm25=bm25,
        all_docs=all_docs,
        k=recall_k,
        verbose=verbose,
        **hybrid_kwargs,
    )

    if not candidates:
        return []

    # 2. 精排：Reranker 打分（可用 ENABLE_RERANK=0 关闭，直接返回粗排 Top-k）
    if not ENABLE_RERANK:
        if verbose:
            print("[Rerank] 已禁用（RAG_ENABLE_RERANK=0），返回粗排 Top-k", file=sys.stderr)
        return candidates[:final_k]

    return rerank(query, candidates, top_k=final_k, verbose=verbose)