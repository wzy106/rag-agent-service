# 五、RAG 完整链路（面试重点）

### 1. 文档加载

```python
docs_dir = Path(__file__).resolve().parent.parent / "docs"
documents = []
# rglob 递归扫描子目录
for md_file in docs_dir.rglob("*.md"):
    text = md_file.read_text(encoding="utf-8")
    rel_path = md_file.relative_to(docs_dir).as_posix()
    documents.append(Document(page_content=text, metadata={"source": rel_path}))
```

- 加载 `docs/` 下**所有子目录**的 Markdown 文件（用 `rglob` 递归）
- **`source` 用相对路径**（如 `notes_interview/q01_xxx.md`），避免不同子目录同名文件冲突
- 当前知识库：98 篇 markdown，切成约 600 块（精确值以索引为准）

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
- 索引文件提交进 Git（约 600 chunk，约 1~2 MB）

### 5. 检索策略

> **实现位置（2026-10-09 起）**：下面所有检索函数在 `rag_core/search.py`，服务端 `mcp_rag_server.py` 和评估脚本 `eval_retrieval.py` 都从这里 import，保证“评估跑的就是部署的”。

#### 5.1 相似度检索

```python
results = vectorstore.similarity_search(query, k=3)
```

最基础的余弦相似度检索。**v3 测试集上 chunk any 48.7%（source strict 62.5%）。**

#### 5.2 MMR 检索

```python
results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
```

- **MMR（最大边际相关性）**：兼顾相关性和多样性
- 公式：`λ × 相关性 - (1-λ) × 与已选 chunk 的最大相似度`（默认 λ=0.5）
- **chunk 级别多样性**，不是文档级别
- 同一文档里差异大的 chunk，MMR 也可能都选
- 想保证“Top-3 来自不同文档”，需要额外按文档去重

**v3 测试集上的表现**：chunk any 只有 **36.0%**（低于相似度 48.7%）——在“精确匹配”任务上吃亏，但多样性较高（2.79 vs 2.52）。

#### 5.3 混合检索（加权 RRF）

**为什么需要**：纯向量检索对关键词不敏感，用户用口语化表达时和文档术语之间有语义鸿沟。

**实现**（`rag_core/search.py`）：

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

**历史表现（v2 source 级口径）**：混合检索召回率 54.6% > 纯相似度 52.2%。v3 换成 source/chunk 双粒度 + 多金标后，具体数字见根目录 `README.md`。

#### 5.4 Rerank 精排

**为什么需要**：混合检索是“双塔 + 关键词”的粗排，速度快但精度有限。Rerank 用交叉编码器把 Top-20 精排到 Top-3，精度更高。

**双塔 vs 交叉编码器**：

| 类型 | 代表模型 | 输入 | 速度 | 精度 |
|---|---|---|---|---|
| 双塔（Embedding） | bge-small-zh-v1.5 | 单条文本 | 快（毫秒级） | 粗 |
| 交叉编码器（Rerank） | bge-reranker-base | **(query, doc) 对** | 慢（CPU 上秒级） | 精 |

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

**效果**：

- 当前评估同时看 candidate recall@20 和 final recall@3，以及 source/chunk 双粒度指标。
- 具体数字见根目录 `README.md` 和 `eval/output/retrieval_v3.json`。
- 定性结论：Rerank 主要提升 final 阶段召回，把 Top-20 里被埋没的相关 chunk 顶上来；如果 candidate recall 已经很高而 final 很低，瓶颈就在精排。

**懒加载设计**：Reranker 模型约 1.1 GB（278M 参数），加载要数秒。用全局变量 `_reranker = None` 做单例，第一次调用 `rerank()` 时才加载，**不拖慢服务启动**（服务启动时另有预热把加载成本前移）。

**性能问题**：Rerank 要逐对推理。267 条查询 × 20 候选 = 5340 次推理，CPU 上精排 20 候选约 2~7 秒（本机实测），267 条跑完约 10~30 分钟。高并发场景需要优化（更小模型 / GPU / ONNX 量化）。

> **踩坑订正**：文档早先写的"单次 50~100 ms"**从未实测**，且与"249 条 4~8 分钟"（≈1~2 s/query）自相矛盾。

#### 5.5 降级策略（避免幻觉）

**问题**：检索不到相关内容时，如果直接调 LLM，模型可能基于无关片段产生幻觉。

**方案**：在检索层加距离阈值，超过就返回“没找到”。

```python
# 通过 eval/calibrate_distance.py 标定
MAX_DISTANCE = 0.96
```

**阈值标定实验**：

- 正样本从测试集抽样；负样本分 easy（`negatives.json`）和 hard（`negatives_hard.json`）
- 对每条问题算 `best_distance`（top-1 的 L2 距离）
- 扫 0.50~1.50 的阈值，画 PR 曲线
- 分别报告 easy / hard / combined 的 FPR

**口径修正**：

- `FPR = FP/(FP+TN)`：负样本被接受的比例
- `FDR = FP/(TP+FP)`：被接受结果里负样本占的比例，依赖抽样基率
- 旧文档把 `1 - precision` 写成“幻觉率 1%”，这个说法不准确，已修正
- 阈值衡量的是检索层接受率，**不是端到端幻觉率**

**已知限制**：

- 只看向量 top-1，且在 BM25/RRF 融合之前判断；BM25 能命中的关键词 query 可能被提前拒绝
- 近域但库内没有的问题可能通过阈值，进入生成阶段
- 真正的拒答需要在生成层做充分性判断或 graph 分支

**产出**：
- `eval/calibrate_distance.py`（标定脚本，可复现）
- `eval/negatives.json` / `eval/negatives_hard.json`
- `eval/output/calibration_v3.json`
- `eval/output/distance_distribution.png` / `pr_curve.png`

**为什么用向量距离，不用 RRF 分数**：
- RRF 分数范围窄（0.005~0.02），无法区分好坏
- 向量 L2 距离直接反映语义相似度，范围 0~2，好判断
- 但 top-1 单点判断仍然脆弱，后续应改成融合后 + 充分性判断
