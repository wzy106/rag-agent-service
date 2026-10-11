# 五、RAG 完整链路（面试重点）

### 1. 文档加载

```python
docs_dir = Path(__file__).resolve().parent.parent / "docs"
for md_file in docs_dir.glob("*.md"):
    text = md_file.read_text(encoding="utf-8")
    documents.append(Document(page_content=text, metadata={"source": md_file.name}))
```

- 加载 `docs/` 下所有 Markdown 文件
- 每个文件转成 `Document` 对象，带 `source` 元数据

### 2. 文档切分

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(documents)
```

- `chunk_size=300`：每块最多 300 字符
- `chunk_overlap=50`：相邻块重叠 50 字符，防止关键信息被切断
- `separators`：按优先级尝试切分，先段落再句子最后字符
- 中文场景专门加了 `。！？` 分隔符
- 当前知识库：2 个 Markdown 文件（真实笔记），切成 102 块

### 3. Embedding

```python
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()
```

- 用本地 `BAAI/bge-small-zh-v1.5` 模型，512 维
- **双塔模型**：query 和 doc 分别编码成向量，算余弦相似度
- 封装成 LangChain 的 `Embeddings` 接口
- 完全离线，不消耗 API
- **HF 缓存路径由 `HF_HOME` 环境变量控制**（三级探测，见面试题 22）

### 4. 向量库 + 持久化

> **变更（2026-10-08）**：`faiss_index/` 已从 `.gitignore` 移除，索引文件（index.faiss 204KB + index.pkl 44KB）已提交进 Git。别人 clone 后可直接构建，无需先跑建索引脚本。

```python
INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

if INDEX_DIR.exists():
    vectorstore = FAISS.load_local(str(INDEX_DIR), embedding, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(str(INDEX_DIR))
```

- 首次启动：构建索引并保存到磁盘（`index.faiss` + `index.pkl`）
- 后续启动：直接从磁盘加载，避免重复 embedding
- 索引文件提交进 Git（248 KB，很小）

### 5. 检索策略

> **实现位置（2026-10-09 起）**：下面所有检索函数在 `rag_core/search.py`，服务端 `mcp_rag_server.py` 和评估脚本 `eval_retrieval.py` 都从这里 import，保证"评估跑的就是部署的"。

#### 5.1 相似度检索

```python
results = vectorstore.similarity_search(query, k=3)
```

最基础的余弦相似度检索。

#### 5.2 MMR 检索

```python
results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
```

- **MMR（最大边际相关性）**：兼顾相关性和多样性
- 公式：`λ × 相关性 - (1-λ) × 与已选 chunk 的最大相似度`（默认 λ=0.5）
- **chunk 级别多样性**，不是文档级别
- 同一文档里差异大的 chunk，MMR 也可能都选
- 想保证"Top-3 来自不同文档"，需要额外按文档去重

#### 5.3 混合检索（加权 RRF）

**为什么需要**：纯向量检索对关键词不敏感，用户用口语化表达时和文档术语之间有语义鸿沟。

**BM25 索引构建**：

```python
import jieba
from rank_bm25 import BM25Okapi

all_docs = list(vectorstore.docstore._dict.values())
tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)
```

**加权 RRF 合并**：

```python
def hybrid_search(query, vectorstore, bm25, all_docs, k=3, rrf_k=60,
                  vector_weight=0.7, bm25_weight=0.3, max_distance=MAX_DISTANCE):
    # 向量检索（带距离分数）
    vector_results_with_scores = vectorstore.similarity_search_with_score(query, k=k*3)

    if not vector_results_with_scores:
        return []

    # 降级判断：最佳向量距离太大 → 认为知识库中无相关内容
    best_distance = vector_results_with_scores[0][1]
    if best_distance > max_distance:
        return []

    # 拆出文档
    vector_results = [doc for doc, _ in vector_results_with_scores]
    vector_ranks = {doc.page_content: i for i, doc in enumerate(vector_results)}

    # BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*3]
    bm25_ranks = {all_docs[i].page_content: rank for rank, i in enumerate(bm25_top_indices)}

    # 加权 RRF
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
```

**RRF 公式**：
```
RRF_score(doc) = Σ 权重 / (k + 排名)
```

- `k=60` 是平滑常数
- 权重控制两路的影响
- 排名越靠前分数越高
- 两路都出现的文档，分数叠加

**加权 RRF 的调优**：
- 等权（0.5/0.5）：97% 召回率（反而变差）
- 加权（0.7/0.3）：100% 召回率
- **向量检索更可靠，给它更高权重**

#### 5.4 Rerank 精排（2026-10-11 新增）

**为什么需要**：混合检索是"双塔 + 关键词"的粗排，速度快但精度有限。Rerank 用交叉编码器把 Top-20 精排到 Top-3，精度更高。

**双塔 vs 交叉编码器**：

| 类型 | 代表模型 | 输入 | 速度 | 精度 |
|---|---|---|---|---|
| 双塔（Embedding） | bge-small-zh-v1.5 | 单条文本 | 快（毫秒级） | 粗 |
| 交叉编码器（Rerank） | bge-reranker-base | **(query, doc) 对** | 慢（百毫秒级） | 精 |

**实现**：

```python
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"
_reranker = None  # 懒加载

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker

def rerank(query, docs, top_k=3):
    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def hybrid_search_with_rerank(query, vectorstore, bm25, all_docs,
                               recall_k=20, final_k=3, **hybrid_kwargs):
    candidates = hybrid_search(query, vectorstore, bm25, all_docs, k=recall_k, **hybrid_kwargs)
    if not candidates:
        return []
    return rerank(query, candidates, top_k=final_k)
```

**效果**（30 条测试项）：

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 混合检索 | 96.7% | 0.86 | 0.94 | 1.30 |
| **混合检索 + Rerank** | 96.7% | **0.90** | **0.97** | 1.17 |

- 精确率 **+0.04**，MRR **+0.03**
- 召回率不变（Rerank 不解决"召回阶段漏掉"的问题）
- 多样性略降（Rerank 偏向最相关，牺牲多样性）——**符合预期**

**懒加载设计**：Reranker 模型 3.36 GB，加载要 10~20 秒。用全局变量 `_reranker = None` 做单例，第一次调用 `rerank()` 时才加载，之后复用。**不拖慢服务启动**。

#### 5.5 降级策略（避免幻觉）

**问题**：检索不到相关内容时，如果直接调 LLM，模型可能基于无关片段产生幻觉。

**方案**：在检索层加距离阈值，超过就返回"没找到"。

```python
# 通过 eval/calibrate_distance.py 标定（2026-10-10）：
# Precision=0.967, Recall=0.967, F1=0.967
# 相比原值 1.1（P=0.909），幻觉率从 9.1% 降到 3.3%
MAX_DISTANCE = 1.0
```

`search_knowledge` 工具处理空结果：

```python
@mcp.tool
def search_knowledge(query: str) -> str:
    results = hybrid_search_with_rerank(query, ..., recall_k=20, final_k=3, verbose=True)
    if not results:
        return "知识库中没有找到相关信息。"
    return "\n\n".join([f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in results])
```

**阈值标定实验（2026-10-10）**：

- 正样本 30 条（来自 `test_set.json`）+ 负样本 25 条（`negatives.json`）
- 对每条问题算 `best_distance`（top-1 的 L2 距离）
- 距离分布：
  - 正样本：0.375 ~ 1.059（中位数 0.698）
  - 负样本：0.854 ~ 1.468（中位数 1.271）
  - **重叠区：[0.854, 1.059]**，宽 0.206

**候选阈值对比**：

| 阈值 | Precision | Recall | F1 | 业务含义 |
|---|---|---|---|---|
| 1.10（原值） | 0.909 | 1.000 | 0.952 | 幻觉率 9.1% |
| 1.06（F1 最大） | 0.938 | 1.000 | 0.968 | 幻觉率 6.2% |
| **1.00（最终）** | **0.967** | **0.967** | **0.967** | **幻觉率 3.3%，漏答 3.3%** |

**为什么选 1.00 而不是 F1 最大的 1.06**：
- RAG 场景用户对"编造"的容忍度远低于"不知道"
- 用 3.3% 的漏答换幻觉率**减半**（9.1% → 3.3%）
- 两个阈值 F1 一样（0.967），但业务上 1.00 更稳

**产出**：
- `eval/calibrate_distance.py`（标定脚本，可复现）
- `eval/negatives.json`（25 条负样本）
- `eval/output/distance_distribution.png`（距离分布图）
- `eval/output/pr_curve.png`（PR 曲线）

**为什么用向量距离，不用 RRF 分数**：
- RRF 分数范围窄（0.005~0.02），无法区分好坏
- 向量 L2 距离直接反映语义相似度，范围 0~2，好判断
