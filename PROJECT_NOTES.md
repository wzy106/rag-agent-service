# RAG Agent 项目复盘笔记

> 用途：面试前自用，把项目讲清楚、把问题想明白。
> 面试话术见 `LESSONS_LEARNED.md`
> 最后更新：2026-10-14（加 Redis 限流；语料 98 篇 / 583 chunk；MAX_DISTANCE 0.96；随机基线 3.0%）

## 一、项目简介（30 秒版本）

基于 LangGraph + MCP + RAG 的企业级 Agent 服务，用 FastAPI 提供 HTTP 接口，支持 SSE 流式输出，API Key 鉴权，Redis 限流，Docker 容器化部署。检索用"混合检索 + Rerank"两阶段 pipeline，知识库 98 篇文档 / 583 chunk，测试集 249 条。

## 二、架构图

```text
客户端（浏览器 / curl / 前端）
    ↓ HTTP POST /chat/stream  （带 Bearer token）
FastAPI（api_agent.py）
    ↓ verify_key 鉴权
    ↓ rate_limit 限流检查
    ↓ agent.astream()
┌──────────────────────────────────┐
│  Agent 循环（LangGraph 状态图）    │
│   ① 调 DeepSeek 大模型            │
│   ② 判断：需要调工具吗？           │
│   ③ 是 → 调 MCP Client            │
│   ④ 拿回工具结果                  │
│   ⑤ 回到 ①                       │
│   若不需要调工具 → 输出最终回答     │
└──────────────────┬───────────────┘
                   │ 工具调用
                   ↓
    MCP Client（MultiServerMCPClient）
                   │ stdio 跨进程
                   ↓
    MCP Server（mcp_rag_server.py）
                   │ search_knowledge
                   ↓
    ┌─────────────────────────────────┐
    │  rag_core/search.py（公共模块）   │
    │       ↓                           │
    │  降级判断（MAX_DISTANCE=0.96）     │
    │       ↓ 通过                      │
    │  混合检索召回 Top-20：             │
    │  向量 Top-20 + BM25 Top-20        │
    │       ↓                           │
    │  加权 RRF 合并（向量0.7+BM25 0.3） │
    │       ↓                           │
    │  Rerank 精排 Top-3                │
    │  （bge-reranker-base, CrossEncoder）│
    └─────────────────────────────────┘
                   ↓
    检索结果返回给 Agent（循环第 ④ 步）
                   ↓ SSE 流式返回
客户端

知识库：
- 98 篇 markdown 文档 / 583 chunk
- 分 5 个子目录：langchain / notes / notes_interview / readme / stories
- 子目录文件数：langchain 12、notes 15、notes_interview 25、readme 13、stories 33

鉴权层：
- Bearer token 通过 Header 传入
- Header + Depends 做依赖注入，鉴权逻辑和业务逻辑解耦
- API_KEY 从环境变量读，代码里不出现真实密钥

限流层：
- Redis 固定窗口限流（incr + expire）
- 按 API Key 限流，默认 20 次/分钟
- Redis 不可用时放行（fail-open），限流不能成为新的单点故障

记忆层：
- LangGraph checkpointer 按 thread_id 管状态（唯一真相源）
- AsyncSqliteSaver 持久化到 SQLite 文件（重启不丢）
- checkpoints.db 挂载到宿主机 ./data/（容器重建也不丢）
- MySQL 只做展示层（给 /history 接口用）

缓存层：
- 暂时没有缓存。历史版本用过 Redis，但命中率结构性为 0，已删除。
```

## 三、技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | LangChain 1.4 + LangGraph 1.2 |
| 模型 | DeepSeek（OpenAI 兼容协议） |
| 工具协议 | MCP（FastMCP 3.4.8） |
| Embedding | bge-small-zh-v1.5（512 维，双塔） |
| Rerank | bge-reranker-base（CrossEncoder，交叉编码器） |
| 检索 | sentence-transformers + FAISS + BM25 + 加权 RRF + Rerank |
| 降级 | 向量距离阈值（标定 0.96，避免幻觉） |
| 鉴权 | Bearer token + FastAPI Depends |
| 限流 | Redis 固定窗口（`incr + expire`），按 API Key |
| 记忆持久化 | AsyncSqliteSaver（生产可换 PostgresSaver） |
| 展示层 | MySQL（对话历史，只读） |
| 服务 | FastAPI + Uvicorn |
| 部署 | Docker + Docker Compose（多阶段构建，2.42 GB） |
| 流式 | SSE（Server-Sent Events） |

## 四、核心功能

- **持久化记忆**：LangGraph checkpointer + AsyncSqliteSaver，重启/容器重建都不丢
- **API Key 鉴权**：`/chat` 和 `/chat/stream` 需要 Bearer token；`Header` + `Depends` 依赖注入
- **Redis 限流**：按 API Key 限流，20 次/分钟（`incr + expire` 固定窗口，fail-open）
- RAG 检索增强（本地 embedding + FAISS 向量库，98 篇 / 583 chunk）
- **两阶段检索**：混合检索召回 Top-20 → Rerank 精排 Top-3
- 混合检索（向量 + BM25 + 加权 RRF）
- **Rerank 精排**（bge-reranker-base，召回率 +4.4%）
- 降级策略（距离阈值标定 0.96，避免幻觉）
- MCP 工具解耦（Agent 和工具分进程）
- SSE 流式输出（打字机效果）
- 中间步骤过滤（只输出最终回答）
- **Agent 护栏**（recursion_limit=10，max_tokens=2000）
- **友好异常处理**（异常类型映射，不泄漏内部信息）
- 请求日志（收到消息 + 耗时统计）
- FAISS 索引持久化（避免重复 embedding）
- MySQL 持久化对话历史（展示层）
- RAG 评估（召回率 + 精确率 + MRR + 多样性 + 随机基线）
- **测试集自动生成**（脚本批量调用 DeepSeek API）
- **阈值标定**（正负样本距离分布 + PR 曲线）
- **Docker 多阶段构建**（镜像从 3.51 GB 降到 2.42 GB）
- **CPU 版 torch**（下载量从 5 GB 降到 600 MB）
- **模型缓存 volume 挂载**（`hf_cache/` 挂宿主机，重建不重下）
- **checkpoints volume 挂载**（`data/` 挂宿主机，容器重建不丢记忆）
- **开发期 volume 挂载**（代码/语料/索引挂载，改代码不用 rebuild）
- **公共检索模块**（`rag_core/search.py`，服务端和评估脚本共用同一份 `hybrid_search`）

## 五、RAG 完整链路（面试重点）

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
- 当前知识库：98 篇 markdown，切成 583 块

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
- 索引文件提交进 Git（583 chunk 约 1~2 MB）

### 5. 检索策略

> **实现位置（2026-10-09 起）**：下面所有检索函数在 `rag_core/search.py`，服务端 `mcp_rag_server.py` 和评估脚本 `eval_retrieval.py` 都从这里 import，保证"评估跑的就是部署的"。

#### 5.1 相似度检索

```python
results = vectorstore.similarity_search(query, k=3)
```

最基础的余弦相似度检索。**在 249 条测试集上召回率 50.6%。**

#### 5.2 MMR 检索

```python
results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
```

- **MMR（最大边际相关性）**：兼顾相关性和多样性
- 公式：`λ × 相关性 - (1-λ) × 与已选 chunk 的最大相似度`（默认 λ=0.5）
- **chunk 级别多样性**，不是文档级别
- 同一文档里差异大的 chunk，MMR 也可能都选
- 想保证"Top-3 来自不同文档"，需要额外按文档去重

**在 249 条测试集上的表现**：召回率只有 **37.8%**——在"精确匹配"任务上吃亏，但多样性较高（2.79）。

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

**在 249 条测试集上的表现**：混合检索召回率 **52.6%** > 纯相似度 50.6%（**+2.0%**）。

#### 5.4 Rerank 精排

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

**效果**（249 条测试项，覆盖 83 篇文档）：

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 混合检索 | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |

- 召回率 **+4.4%**，MRR **+0.05**
- **Rerank 提升的主要是召回率**——混合检索直接返回 Top-3 会漏掉"相关但排名 4~20"的 chunk，Rerank 拿 Top-20 重新排，有机会把关键 chunk 顶上来

**懒加载设计**：Reranker 模型 3.36 GB，加载要 10~20 秒。用全局变量 `_reranker = None` 做单例，第一次调用 `rerank()` 时才加载，**不拖慢服务启动**。

**性能问题**：249 条查询 × 20 候选 = 4980 次推理，CPU 上单次 50~100 ms，总耗时 **4~8 分钟**。高并发场景需要优化（更小模型 / GPU / 缓存）。

#### 5.5 降级策略（避免幻觉）

**问题**：检索不到相关内容时，如果直接调 LLM，模型可能基于无关片段产生幻觉。

**方案**：在检索层加距离阈值，超过就返回"没找到"。

```python
# 通过 eval/calibrate_distance.py 标定（2026-10-12）：
# Precision=0.990, Recall=1.000, F1=0.995
# 相比旧阈值 1.0（P=0.967），精确率提升，负样本误入更少
MAX_DISTANCE = 0.96
```

**阈值标定实验（2026-10-12）**：

- 正样本 100 条（从 `test_set_v2.json` 抽样，seed=42）+ 负样本 25 条（`negatives.json`）
- 对每条问题算 `best_distance`（top-1 的 L2 距离）
- 距离分布：
  - 正样本：0.363 ~ 0.958（中位数 0.709）
  - 负样本：0.813 ~ 1.420（中位数 1.176）
  - **重叠区：[0.813, 0.958]**，宽 0.145

**候选阈值对比**：

| 阈值 | Precision | Recall | F1 | 业务含义 |
|---|---|---|---|---|
| 1.00（旧） | 0.967 | 0.967 | 0.967 | 旧标定 |
| **0.96（最终）** | **0.990** | **1.000** | **0.995** | **幻觉率约 1.0%，漏答 0%** |

**为什么选 0.96 而不是旧 1.00**：
- 正样本距离上界是 0.958，取 0.96 可以全收正样本
- 比 1.0 更紧，能减少负样本误入上下文
- RAG 场景用户对"编造"的容忍度远低于"不知道"，优先保精确率
- 新标定 P=0.990、R=1.000、F1=0.995，整体优于旧阈值

**产出**：
- `eval/calibrate_distance.py`（标定脚本，可复现）
- `eval/negatives.json`（25 条负样本）
- `eval/output/distance_distribution.png`（距离分布图）
- `eval/output/pr_curve.png`（PR 曲线）

**为什么用向量距离，不用 RRF 分数**：
- RRF 分数范围窄（0.005~0.02），无法区分好坏
- 向量 L2 距离直接反映语义相似度，范围 0~2，好判断

## 六、遇到的坑（面试重点）

| 坑 | 原因 | 解决方案 |
|---|---|---|
| DeepSeek 不支持 json_schema | 模型 API 限制 | 改用 PydanticOutputParser，本地解析 |
| 中文引号混入字符串 | 复制时带入全角引号 | 统一用英文半角引号 |
| fastmcp 4.x 与 langchain-mcp-adapters 冲突 | mcp 版本要求矛盾（<2.0 vs >=2.0） | 降级 fastmcp 到 3.4.8 |
| HuggingFace 联网卡住 | 国内无法访问 huggingface.co | 设 HF_HUB_OFFLINE=1，用本地缓存路径 |
| uvicorn --reload 下调 asyncio.run 报错 | 事件循环嵌套 | 改用 FastAPI lifespan 异步钩子 |
| MCP 工具同步调用报错 | MCP 工具是异步的 | Agent 改用 astream / ainvoke |
| stream 输出吞进中间步骤 | 工具调用前的英文思考也被流式输出 | 记录 tool 节点位置，只输出它之后的内容 |
| Docker 构建慢 | 模型下载 + 依赖安装 | 分层缓存；模型下载移出构建期 |
| MCP Server 里的 print 看不到 | MCP 协议占用 stdout | 改成输出到 stderr |
| 等权混合检索反而变差 | RRF 奖励两路都出现的文档，BM25 判断错会被放大 | 加权 RRF（向量 0.7 / BM25 0.3） |
| 多轮对话缓存串台 | 同一问题在不同上下文答案不同 | 缓存 key 加 thread_id |
| 缓存命中率结构性为 0 | context_version 每轮 +2，同一问题第二次必然 miss | **删掉缓存**，承认是负优化 |
| 无关问题触发幻觉 | 检索无相关内容时直接调 LLM | 距离阈值降级，返回"没找到"不调 LLM |
| 单一评估指标误导 | 只看召回率无法区分策略 | 加精确率、MRR、多样性 |
| 双源记忆打架 | InMemorySaver + MySQL 手动拼历史 | 统一到 LangGraph checkpointer（AsyncSqliteSaver） |
| **错误信息泄漏** | `str(e)` 直接返回客户端，可能泄漏 API key / 路径 | 映射成 `friendly_error` 友好提示 |
| **friendly_error 子串误判** | `"rate" in msg` 会匹配 generate / operate | 改用异常类型 + HTTP 状态码判断 |
| **finally 里 yield 报错** | 客户端断连时 `aclose()` 抛 GeneratorExit | `[DONE]` 移出 finally，加 `except asyncio.CancelledError` |
| **requirements.txt 缺依赖** | 逐个 pip install，忘了同步回清单 | 干净 venv 里 `pip install -r requirements.txt` 验证 |
| **Docker 构建 `unpigz corrupted`** | C 盘 0 字节，镜像层解压时写入失败 | 清理 C 盘 → Docker 数据迁到 D 盘 |
| **Docker 数据占满 C 盘 57 GB** | WSL2 vhdx 默认存在 C 盘 | Settings → Resources → Disk image location 改到 `D:\Docker` |
| **pip 清华源 ReadTimeout** | 容器内下载 CUDA 全家桶（triton 248MB、cudnn 500MB、nccl 200MB） | 换阿里云源 + 长超时 + 重试；再用 CPU 版 torch 砍掉 CUDA 全家桶 |
| **Windows wheel 不能给 Linux 容器用** | 宿主机是 `win_amd64` + `cp314`，容器是 `manylinux` + `cp312` | wheelhouse 作废，改用容器内直接下载 CPU 版 |
| **MCP Server 硬编码 HF 缓存路径** | 代码写死 `Path.home()/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots` | 改成三级探测（环境变量 > 容器默认挂载点 > 项目 hf_cache） |
| **checkpoints.db 无 volume** | DB 存在容器内部，`docker compose down` 后记忆全丢 | compose 加 `./data:/app/data`，代码用 `DATA_DIR` 环境变量 |
| **评估脚本和服务端代码分叉** | 两份 `hybrid_search`，参数还不一样 | 抽到 `rag_core/search.py`，两边都 import |
| **MAX_DISTANCE 拍脑袋定** | 原值 1.1 是跑了几条样例的约数，无统计支撑 | `eval/calibrate_distance.py` 标定 → 0.96（P=0.990, R=1.000, F1=0.995；正样本 100 条） |
| **Rerank 收益被语料规模压制** | 旧语料只有 2 篇 102 chunk，天花板效应 | 扩语料到 98 篇 583 chunk，Rerank 增益稳定为召回 +4.4%、MRR +0.05 |
| **测试集太小导致判别力不足** | 30 条测试项，随机基线 84.1%，判别空间只有 15.9 个点 | 扩到 249 条 + 98 篇文档，随机基线降到 3.0%，判别空间 54.0 个点 |
| **拆笔记文件 chunk 太集中** | `notes/07` 一个文件 74 chunks（含 25 道面试题） | 按 `###` 拆成 25 个独立文件，判别力提升 |
| **接口裸奔** | 一开始没加鉴权，任何人都能调 | Bearer token + FastAPI `Depends` 依赖注入；`API_KEY` 从环境变量读 |
| **改代码每次 rebuild** | 代码/语料通过 COPY 打进镜像，改一行就要 rebuild | 开发期 volume 挂载，改代码 `docker compose restart` 即可；收尾时改回 COPY |
| **限流未做** | Redis 依赖一直是"预留"，从没真正用过 | 按 API Key 限流（`incr + expire`），fail-open |

## 七、面试可能被问的问题（附答案要点）

### 1. 为什么用 MCP，而不是普通 @tool？

- 普通 @tool：工具和 Agent 同进程，硬编码，只能自己用
- MCP：工具独立成 Server，跨进程通信，任何支持 MCP 的 Client 都能调用
- 好处：解耦、可复用、可独立部署、可跨语言

### 2. RAG 的完整流程？

1. 文档加载（Document）
2. 切分（chunking）
3. Embedding（向量化）
4. 存入向量库
5. 用户提问 → 向量化 → 相似度检索 Top-K
6. 检索结果拼进 Prompt
7. 模型生成回答

**进阶版（你项目用的）**：
1. 文档加载 → 2. 切分 → 3. Embedding → 4. 存入 FAISS
5. 用户提问 → **混合检索召回 Top-20（向量 + BM25 + RRF）**
6. **Rerank 精排 Top-3（CrossEncoder）**
7. **降级判断（距离阈值 0.96）**
8. 检索结果拼进 Prompt → 模型生成回答

### 3. 怎么防止 RAG 幻觉？

**两层防护 + Rerank**：

**第一层（Prompt 软约束）**：
- Prompt 里加"只使用资料中的信息，不要编造"

**第二层（检索层硬降级）**：
- 用向量距离做阈值（标定 0.96），超过就返回"知识库中没有相关信息"
- 不调 LLM，从根源上避免幻觉

**加分项（精排）**：
- 混合检索的 Top-20 用 Reranker 精排，把不相关的候选挤下去
- 召回率 52.6% → 57.0%

### 4. InMemorySaver 和 PostgresSaver 区别？

- InMemorySaver：存内存，进程结束即丢失，适合开发测试
- PostgresSaver：存数据库，持久化，多实例共享，适合生产
- **本项目用 AsyncSqliteSaver**：存 SQLite 文件，持久化，单机部署方便

### 5. 流式输出怎么实现的？

- Agent 用 `astream(..., stream_mode="messages")`
- 每拿到一个 token chunk，用 SSE 格式 `data: {json}\n\n` 发给客户端
- 客户端用 EventSource / fetch reader 逐块接收并渲染
- 最后发 `data: [DONE]\n\n` 结束

### 6. 为什么要用 Docker？

- 环境一致性：换机器不用担心依赖问题
- 隔离：不污染宿主机环境
- 部署简单：一条 `docker compose up -d`
- 可复现：镜像=代码+环境+依赖

**追问：你的 Docker 镜像多大？怎么优化的？**

> "从 3.51 GB 优化到 2.42 GB。三个关键点：
> ① **多阶段构建**：builder 阶段装 build-essential（编译器 300MB+），运行时不带，只 `COPY --from=builder /install /usr/local`；
> ② **CPU 版 torch**：原来默认装 CUDA 版，会拉 triton 248MB + cudnn 500MB + nccl 200MB + nvshmem 100MB，换成 `--extra-index-url https://download.pytorch.org/whl/cpu` 后，bge-small 这种小模型在 CPU 上毫秒级，根本用不上 GPU；
> ③ **模型下载移出构建期**：原来 Dockerfile 里有一行 `RUN python -c "...下载 embedding 模型..."`，构建耗时 33 分钟且容易网络超时。改成运行时 volume 挂载 `hf_cache`，构建期不下载。"

### 7. 并发量高怎么优化？

- Agent 改用异步（astream / ainvoke）
- MCP Server 常驻，不用每次启动
- 向量库换 FAISS / Chroma 持久化版本
- **加 Redis 限流（incr + expire）—— 已落地**
- 多 worker 部署（uvicorn --workers N 或 gunicorn）
- 反向代理（Nginx）分担负载
- **Reranker 是瓶颈**：单条 query 精排约 50~100 ms（CPU），249 条需 4~8 分钟；高并发可换更小 Reranker 或加缓存

### 8. 文档切分参数怎么定？

- `chunk_size=300`：中文短句语义密度高，300 字符约一段
- `chunk_overlap=50`：约为 chunk_size 的 15%，防止边界信息丢失
- `separators`：中文加 `。！？`，优先在段落边界切

### 9. MMR 是什么？

**全称**：Maximal Marginal Relevance（最大边际相关性）。

**公式**：
```
MMR = λ × Sim(query, chunk) - (1-λ) × max(Sim(chunk, 已选chunk))
```

**关键理解**：
- MMR 在 **chunk 级别**做多样性，不是文档级别
- 它避免的是"内容重复"，不是"来源重复"
- 同一个文档里的两个差异大的 chunk，MMR 也会都选
- 想保证"Top-3 来自不同文档"，需要额外按文档去重

**在 249 条测试集上的表现**：
- MMR 召回率只有 37.8%（低于相似度 50.6%）——**在"精确匹配"任务上吃亏**
- 多样性 2.79（高于相似度 2.63）—— **多样性导向是对的**

### 10. 混合检索是什么？

**全称**：Hybrid Search。

**组合方式**：
- 向量检索：按语义相似度找
- BM25：按关键词精确匹配找
- 两路并行，结果用 RRF 合并

**RRF 公式**：
```
RRF_score(doc) = Σ 权重 / (k + 排名)
```

**关键理解**：
- RRF 奖励"两路都出现"的文档
- 如果 BM25 判断错，错误会被放大
- 用加权 RRF 让更可靠的一路权重更高（我用向量 0.7 / BM25 0.3）

**在 249 条测试集上的表现**：
- 混合检索 52.6% > 纯相似度 50.6%（**+2.0%**）
- 说明 BM25 补的关键词匹配确实有用

### 11. 怎么防止 Agent 死循环？

**两道护栏**：

1. **`recursion_limit=10`**：Agent 最多循环 10 次，超过抛异常
2. **`max_tokens=2000`**：单次输出最多 2000 token

**触发后怎么办**：异常被 `try/except` 捕获，以 SSE 格式返回友好错误，服务不崩。

**面试话术**：
> "我给 Agent 设了两道护栏：一是 recursion_limit=10，超过自动抛异常终止；二是 max_tokens=2000，限制单次输出。第一道防止 Agent 无限调用工具，第二道防止单次回答过长。"

### 12. Agent 多轮对话怎么实现？

**方案**：LangGraph checkpointer + thread_id。

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(data_dir / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        agent = create_agent(model, tools=tools, checkpointer=saver, system_prompt="...")
        yield

# 调用
config = {"configurable": {"thread_id": thread_id}}
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": payload.message}]},
    config=config
)
```

**关键点**：

| 概念 | 作用 |
|---|---|
| `AsyncSqliteSaver` | 持久化 checkpointer，存 SQLite 文件 |
| `checkpointer=saver` | 把存储器挂到 Agent |
| `thread_id` | 区分不同会话，相同 ID 共享历史 |
| `DATA_DIR` | 决定 DB 路径，本地/容器两套值 |

**测试验证**：
1. `test_001` 问"MMR 是什么？" → 正常回答
2. **`docker compose down` 完全销毁容器**
3. `docker compose up -d` 重建
4. `test_001` 问"它和普通相似度检索有什么区别？" → 仍能理解"它"指 MMR ✅

**面试话术**：
> "我的记忆用 LangGraph 的 checkpointer 做唯一真相源，持久化用 AsyncSqliteSaver（生产环境会换 PostgresSaver）。每次请求只传当前这一轮的消息，Agent 自动从 checkpointer 加载该 thread_id 的历史。之前我试过手动从 MySQL 加载历史拼进 messages，但发现和 checkpointer 的 state 会打架——消息格式混用、历史可能重复累积。后来统一到 checkpointer，MySQL 只做展示层给 /history 接口用。另外我把 checkpoints.db 挂到宿主机的 ./data/，这样 docker compose down 后容器重建也不丢。"

### 13. 检索不到内容怎么处理？

**降级策略**：

1. **检索层**：用向量距离做阈值（`MAX_DISTANCE = 0.96`，标定得出），超过就返回空
2. **工具层**：收到空结果时返回"知识库中没有找到相关信息"
3. **不调 LLM**：从根源上避免模型基于无关片段产生幻觉

**阈值怎么定**：
- 标定实验：正样本 100 条 + 负样本 25 条
- 距离分布：正 0.363~0.958，负 0.813~1.420
- 选 0.96：P=0.990, R=1.000, F1=0.995（详见面试题 24）

**面试话术**：
> "我在检索层加了降级策略。用相似度距离做阈值，超过阈值就返回'知识库中没有相关信息'，不调 LLM。这样避免模型在缺乏依据时产生幻觉。阈值不是拍脑袋定的——我构造了 100 条正样本和 25 条负样本做标定实验，最终取 0.96，精确率 99.0%，召回率 100%，幻觉率约 1.0%，漏答 0%。"

### 14. RAG 评估用什么指标？

**四个指标 + 随机基线**：

| 指标 | 公式 | 含义 |
|---|---|---|
| 召回率 | 命中数 / 总数 | 该找的有没有找到 |
| 精确率 | 平均（相关结果数 / Top-K 总数） | 找到的有多少是对的 |
| MRR | 平均（1 / 第一个正确结果的排名） | 第一个正确结果排多前 |
| 多样性 | 平均（Top-K 里不同文档数） | 结果覆盖多少不同来源 |
| 随机基线 | 1 - C(N-n,k)/C(N,k) | 从 N 个 chunk 里随机抽 k 个，命中期望 source 的概率 |

**实测数据**（249 个测试项，覆盖 83 篇文档）：

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 普通相似度 | 50.6% | 0.23 | 0.39 | 2.63 |
| MMR | 37.8% | 0.15 | 0.33 | 2.79 |
| MMR + 去重 | 40.6% | 0.14 | 0.34 | **3.00** |
| 混合检索 | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |
| **随机基线** | **3.0%** | — | — | — |

**面试话术**：
> "我评估 RAG 用了四个指标 + 随机基线。随机基线用组合公式精确计算——从 583 个 chunk 里随机抽 3 个，命中期望 source 的概率是 3.0%。这里有个关键：**早期版本语料只有 2 篇文档、测试集只有 30 条，随机基线高达 84.1%，判别空间只有 15.9 个点**——Rerank 的提升被天花板效应压制，只有 +0.04 精确率。**扩语料到 98 篇、测试集到 249 条后**，随机基线降到 3.0%，判别空间 54.0 个点，**Rerank 的增益变成召回率 +4.4%、MRR +0.05**——这才是真实的增益。"

### 15. 错误信息怎么处理？

**问题**：直接返回 `str(e)` 可能泄漏 API key、文件路径、SQL 结构等内部信息。

**方案**：用 `friendly_error` 映射成用户可读的提示。**用异常类型 + HTTP 状态码判断，不做自然语言子串匹配。**

```python
def friendly_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    if status == 401:
        return "模型服务认证失败，请联系管理员"
    if status == 429:
        return "请求过于频繁，请稍后再试"
    if isinstance(e, asyncio.TimeoutError):
        return "请求超时，请稍后再试"
    if isinstance(e, pymysql.MySQLError):
        return "存储服务暂时不可用，请稍后再试"
    return "服务暂时不可用，请稍后再试"
```

**为什么不用子串匹配**：`"rate" in msg` 会误匹配 `generate`、`operate`、`separate` 等词，导致错误分类。

### 16. 如果让你重做，会改什么？

- 用 uv + 虚拟环境管理依赖，避免全局冲突
- 加 LangSmith 可观测性
- 写单元测试和集成测试
- **评估升级到 chunk 级标注**（现在是文档级）
- **加端到端评估**（LLM-as-judge 或人工标注）

### 17. Docker 构建失败怎么排查？

**案发现场**：
```
#16 exporting to image
#16 ERROR: failed to extract layer sha256:f7df...: 
    exit status 1: unpigz: skipping: <stdin>: corrupted -- incomplete deflate data
failed to solve: Unavailable: error reading from server: EOF
```

**排查思路**：
1. **看错误阶段**：报在 `exporting to image` / `unpacking` → 是写盘阶段，不是编译阶段
2. **查磁盘空间**：`Get-PSDrive C` → 显示 Free = 0.00 GB ✅ 命中
3. **查 Docker 数据位置**：`wsl -d docker-desktop df -h` → `/mnt/host/c` 100% 满

**根因**：C 盘 200 GB 用满，Docker 解压镜像层时写不进去，管道被截断。

**解法**：
1. 清 C 盘（cleanmgr + 手动删 Temp/SoftwareDistribution/HF cache）
2. Docker 数据迁到 D 盘（Settings → Disk image location）
3. C 盘从 10 GB → 65 GB 可用

### 18. 为什么 pip 突然开始超时？

**触发点**：我改 Dockerfile 时，pip 命令加了 `--prefix=/install`，Docker 层缓存失效，必须重新下载所有依赖。

**原来为什么"不超时"**：之前的构建命中旧缓存层，pip 实际没走网络。

**超时原因**：torch 默认拉 CUDA 全家桶，光 triton 就 248 MB，加上 cudnn (500 MB) + nccl (200 MB) + nvshmem (100 MB)，总量 3~5 GB。清华源大包容易超时。

**解法**：
- 换阿里云源（比清华稳定）
- 加 `--timeout 300 --retries 10`
- **根本解法**：requirements.txt 加 `--extra-index-url https://download.pytorch.org/whl/cpu`，用 CPU 版 torch，下载量从 5 GB → 600 MB

### 19. Windows 下载的 wheel 能给 Linux 容器用吗？

**不能。** 三个维度都必须匹配：

| 维度 | Windows 宿主机 | Linux 容器 |
|---|---|---|
| 操作系统 | `win_amd64` | `manylinux*` |
| Python 版本 | `cp314`（本机 3.14） | `cp312`（容器 3.12） |
| ABI | Windows ABI | Linux ABI |

**验证方法**：看 wheel 文件名后缀。
- `torch-2.14.1+cpu-cp314-cp314-win_amd64.whl` ❌ 容器用不了
- `torch-2.x.x+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` ✅ 容器能用

**教训**：**别在宿主机给容器下 wheel**，除非用 `pip download --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:` 明确指定平台。

### 20. 多阶段构建的好处是什么？

```dockerfile
# Stage 1: builder
FROM python:3.12-slim AS builder
RUN apt-get install build-essential   # 编译器，300MB+
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: runtime
FROM python:3.12-slim
COPY --from=builder /install /usr/local   # 只拷依赖，不拷编译器
```

**好处**：
- 运行时镜像**不含 build-essential**（编译器 + 头文件 ~300 MB）
- 减少攻击面（没有 gcc/g++ 可被利用）
- 镜像更小（3.51 GB → 2.42 GB）

### 21. Docker 容器重建后，LangGraph 的记忆会丢吗？

**会，如果不做持久化。**

`AsyncSqliteSaver` 默认把 `checkpoints.db` 写到容器内部，`docker compose down` 一销毁就全没了。README 里说"服务重启不丢历史"，对**进程重启**成立，对**容器重建**不成立。

**修法**：把 checkpoints 目录挂载到宿主机。

```python
# api_agent.py：用 DATA_DIR 环境变量决定路径
data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
data_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = str(data_dir / "checkpoints.db")
async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
    ...
```

```yaml
# docker-compose.yml：加 volume + 环境变量
environment:
  - DATA_DIR=/app/data
volumes:
  - ./data:/app/data
```

**验证方法**：
1. 问一个问题
2. `docker compose down`（完全销毁容器）
3. `docker compose up -d`
4. 用代词问 → 能理解"它"指上一轮 MMR ✅

**为什么不直接挂载 `./checkpoints.db:/app/rag_api/checkpoints.db`**：如果宿主机文件不存在，Docker 会**创建一个同名目录**挂进去，导致 SQLite 打不开。挂目录比挂文件稳。

### 22. 为什么要让 HF_HOME 三级探测？

**问题**：MCP Server 是 `api_agent.py` 用 stdio 拉起的**子进程**，`MultiServerMCPClient` 起子进程时环境变量可能不完整传递，导致容器里 `HF_HOME` 为空，代码 fallback 到项目内 `hf_cache/`（容器里不存在），最终 `FileNotFoundError`。

**解法**：

```python
def _resolve_hf_home() -> str:
    if os.environ.get("HF_HOME"):           # 1. 显式环境变量
        return os.environ["HF_HOME"]
    container_default = Path("/root/.cache/huggingface")
    if container_default.exists():          # 2. 容器默认挂载点
        return str(container_default)
    return str(Path(__file__).resolve().parent.parent / "hf_cache")   # 3. 本地开发
```

**为什么不硬编码**：硬编码 `Path.home()/.cache/huggingface/.../snapshots` 依赖精确的目录结构，换环境必崩，且不易发现。

**为什么不只用环境变量**：子进程可能拿不到，本地开发时也常常忘了设。

### 23. 为什么要抽公共 `hybrid_search` 模块？

**问题**：评估脚本和服务端各有一份 `hybrid_search` 实现，参数不一样（评估脚本缺距离降级），所以"评估跑的数字"描述的不是"部署的系统"。

**修法**：抽到 `rag_core/search.py`，服务端和评估脚本都 `from rag_core.search import hybrid_search`。

**Dockerfile 也要加 COPY**：
```dockerfile
COPY rag_core/ ./rag_core/
```

**`mcp_rag_server.py` 加 sys.path hack**（脚本在 `mcp_test/` 子目录，找不到项目根的 `rag_core`）：
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import hybrid_search, ...
```

**收益**：
- 评估结果真实反映部署行为
- 改一处，两端生效
- 面试报的每个数字都能被复现

### 24. `MAX_DISTANCE` 阈值怎么定的？

**不是拍脑袋定的，是标定出来的。**

**方法**：
1. 构造正样本（100 条知识库有问题，从 `test_set_v2.json` 抽样，seed=42）+ 负样本（25 条知识库没有的问题）
2. 对每条问题算 `best_distance`（top-1 的 L2 距离）
3. 画正负样本距离分布直方图，看分离点
4. 扫 101 个候选阈值（0.5~1.5），计算 Precision / Recall / F1

**结果**：

| 阈值 | Precision | Recall | F1 |
|---|---|---|---|
| 1.00（旧） | 0.967 | 0.967 | 0.967 |
| **0.96（当前）** | **0.990** | **1.000** | **0.995** |

**为什么选 0.96 而不是旧 1.00**：
- 正样本距离上界是 0.958，取 0.96 可以全收正样本
- 比 1.0 更紧，能减少负样本误入上下文
- RAG 场景用户对"编造"的容忍度远低于"不知道"
- 新标定 P=0.990、R=1.000、F1=0.995，精确率和召回率都更好

**产出**：
- `eval/calibrate_distance.py`（标定脚本）
- `eval/negatives.json`（25 条负样本）
- `eval/output/distance_distribution.png`（距离分布图）
- `eval/output/pr_curve.png`（PR 曲线）

**面试话术**：
> "MAX_DISTANCE=0.96 不是拍脑袋定的。我构造了 100 条正样本和 25 条负样本，对每条跑 best_distance，画分布直方图看分离点，再扫阈值从 0.5 到 1.5 画 PR 曲线。最终取 0.96——因为这个点上精确率 99.0%，召回率 100%，F1 0.995；正样本距离上界是 0.958，0.96 刚好全收正样本，同时比旧阈值 1.0 更紧，能减少负样本误入。标定脚本和结果图都在仓库里，可以复现。"

### 25. 为什么要加 Rerank？

**问题**：混合检索（向量 + BM25 + RRF）是"双塔 + 关键词"的粗排——快但精度有限。Top-3 里可能混进一些语义上相关但实际帮助不大的 chunk。

**方案**：用 `bge-reranker-base` 交叉编码器对混合检索召回的 Top-20 精排到 Top-3。

**双塔 vs 交叉编码器**：

| 维度 | 双塔（bge-small） | 交叉编码器（bge-reranker） |
|---|---|---|
| 结构 | query 和 doc 分别编码 | (query, doc) 拼一起送进模型 |
| 速度 | 快（毫秒级） | 慢（百毫秒级） |
| 精度 | 粗 | 精 |
| 用途 | 大规模召回 | 小规模精排 |

**关键设计：两阶段检索**
1. **粗排**：混合检索召回 Top-20（保证召回率）
2. **精排**：Reranker 打分保留 Top-3（保证精确率）

**为什么不能只用 Reranker**：要对每条 (query, doc) 跑一次模型。583 个 chunk → 583 次推理，100 ms × 583 ≈ 58 秒/查询——太慢。所以先用双塔快速筛到 20 条，再用 Reranker 精排。

**效果**（249 条测试项，覆盖 83 篇文档）：

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 混合检索 | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |

- 召回率 **+4.4%**
- MRR **+0.05**
- **Rerank 提升的主要是召回率**（把 Top-20 里被埋没的相关 chunk 顶上来）

**诚实的局限**：
- 249 条查询 × 20 候选 = 4980 次推理，CPU 上 50~100 ms/次，总耗时 **4~8 分钟**
- 高并发场景需要优化（更小模型 / GPU / 缓存）

**面试话术**：
> "我加了两阶段检索：混合检索召回 Top-20，用 bge-reranker-base 交叉编码器精排 Top-3。在 249 条测试集上召回率从 52.6% 提到 57.0%，MRR 从 0.42 提到 0.47。Reranker 是懒加载的——第一次调用才加载 3.36 GB 模型，不拖慢服务启动。这里有个诚实的局限：Rerank 单次推理 CPU 上 50~100 ms，249 条查询需要 4~8 分钟，高并发场景需要优化。"

### 26. 为什么要加 API Key 鉴权？

**问题**：`/chat` 和 `/chat/stream` 接口完全裸奔，任何人知道 URL 就能调。

**方案**：Bearer token + FastAPI `Depends` 依赖注入。

```python
API_KEY = os.getenv("API_KEY", "dev-key-change-me")

async def verify_key(authorization: str = Header(default="")) -> str:
    if not authorization:
        raise HTTPException(401, detail="缺少 Authorization 头")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Authorization 格式错误")
    token = authorization[7:].strip()
    if token != API_KEY:
        raise HTTPException(403, detail="API Key 无效")
    return token

@app.post("/chat")
async def chat(payload: ChatRequest, token: str = Depends(verify_key)):
    ...
```

**关键设计**：

| 点 | 说明 |
|---|---|
| `Header` | FastAPI 自动从请求头取值 |
| `Depends` | 依赖注入，接口调用前先跑 `verify_key` |
| `401` vs `403` | 401 = 没带凭证；403 = 带了但不对 |
| `API_KEY` 环境变量 | 密码外置，代码里不出现真实密钥 |
| `${API_KEY:-默认}` | compose 从宿主机读，注入容器 |
| `verify_key` 返回 token | 供后续限流用 |

**面试话术**：
> "接口加了 API Key 鉴权。用 FastAPI 的 `Header` + `Depends` 做依赖注入，`/chat` 和 `/chat/stream` 共用同一个 `verify_key`，鉴权逻辑和业务逻辑解耦。`API_KEY` 从环境变量读——本地 `.env` 一份，生产服务器一份，代码里不出现真实密钥。`docker-compose.yml` 里用 `${API_KEY:-dev-key-change-me}` 语法从宿主机读取，读不到用默认值。这样 yml 文件能安全提交，真实密码留在 `.env` 里。"

### 27. 为什么开发期用 volume 挂载？

**问题**：`api_agent.py` 和 `docs/` 等通过 `COPY` 打进镜像，每次改代码都要 `docker compose down && build && up`，迭代慢，磁盘越吃越多。

**方案**：开发期把频繁变动的目录改成 volume 挂载。

```yaml
volumes:
  - ./rag_api:/app/rag_api
  - ./mcp_test:/app/mcp_test
  - ./rag_core:/app/rag_core
  - ./docs:/app/docs
  - ./faiss_index:/app/faiss_index
```

**关键调整**：`DATA_DIR` 从 `/app/rag_api/data` 改成 `/app/data`（避免和 `./rag_api` 挂载冲突）。

**效果**：

| 改了什么 | 挂载前 | 挂载后 |
|---|---|---|
| 代码 | `down` + `build` + `up` | `docker compose restart agent-api` |
| 语料 | 同上 | restart |
| 时间 | 几秒到几分钟 | 2~3 秒 |
| 磁盘 | 每次涨 | 零增长 |

**收尾时改回 COPY**：项目结束、要交付时，把开发挂载注释掉，重新 build，镜像"冻结"。

**面试话术**：
> "开发期我把代码和语料目录挂载进容器，改代码只要 `docker compose restart`，不用 rebuild。因为项目早期每次改一行都要 build，磁盘也越吃越多——`docker system df` 一看，镜像 28 GB、缓存 11 GB。挂载后改代码 2 秒生效，磁盘零增长。收尾时把挂载注释掉重新 build，镜像就是自包含的，不依赖宿主机文件。开发期追求迭代速度，生产期追求可复现性。"

### 28. Redis 限流怎么做的？

**问题**：加了鉴权只防"谁能调"，不防"调多少次"。一个 Key 可以无限打。

**方案**：固定窗口限流，按 API Key 计，20 次/分钟。

```python
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

def rate_limit(key: str):
    try:
        r = _get_redis()
        redis_key = f"rate:{key}"
        current = r.incr(redis_key)
        if current == 1:
            r.expire(redis_key, 60)   # 第一次访问设置 60 秒过期
        if current > RATE_LIMIT_PER_MINUTE:
            raise HTTPException(429, detail="请求过于频繁...")
    except HTTPException:
        raise
    except redis.RedisError as e:
        print(f"[限流] Redis 不可用，放行：{e}")
```

**接口挂上**：

```python
@app.post("/chat")
async def chat(payload: ChatRequest, token: str = Depends(verify_key)):
    rate_limit(token)
    ...
```

**关键设计**：

| 点 | 说明 |
|---|---|
| 维度 | 按 API Key 限流，复用鉴权拿到的 token |
| 算法 | 固定窗口（`incr + expire`），不做滑动窗口 |
| fail-open | Redis 挂了放行，限流不能成为新单点 |
| 配额 | 环境变量 `RATE_LIMIT_PER_MINUTE`，默认 20 |
| `/history` 不限流 | 展示层流量小 |

**验证**：

```cmd
for /L %i in (1,1,21) do @curl.exe ... -w "%i : %{http_code}\n" ...
:: 前 20 次 200，第 21 次 429
```

Redis 计数器：

```cmd
docker exec -it agent-redis redis-cli get "rate:wzyyqw106"
:: "21"
docker exec -it agent-redis redis-cli ttl "rate:wzyyqw106"
:: 47
```

**面试话术**：
> "我把 Redis 从'预留'变成'落地'——按 API Key 限流，20 次/分钟。用 `incr + expire` 做固定窗口，比滑动窗口简单，够用。关键设计是 **fail-open**：Redis 挂了就放行，限流是保护措施，不能让它成为新的单点故障。鉴权 + 限流是生产环境的两层保护：鉴权防'谁能调'，限流防'调多少次'。"

## 八、项目数据（面试时能报的具体数字）

- 工具数量：1 个（search_knowledge）
- **知识库文档数：98 篇 markdown（5 个子目录）**
- **切分后块数：583 块**
- Embedding 模型：BAAI/bge-small-zh-v1.5（512 维，双塔，192 MB）
- **Rerank 模型：BAAI/bge-reranker-base（CrossEncoder，3.36 GB）**
- 向量库：FAISS（本地持久化）
- 检索策略：**两阶段检索**（混合检索召回 Top-20 + Rerank 精排 Top-3）
- RRF 参数：k=60, vector_weight=0.7, bm25_weight=0.3
- 降级阈值：MAX_DISTANCE=0.96（标定得出，P=0.990, R=1.000, F1=0.995）
- 幻觉率：约 1.0%，漏答 0%（新标定）
- 护栏参数：recursion_limit=10, max_tokens=2000
- 鉴权：Bearer token（`/chat` 和 `/chat/stream`），`API_KEY` 从环境变量读
- 限流：Redis 固定窗口（`incr + expire`），按 API Key 限流 20 次/分钟，fail-open
- 记忆：LangGraph checkpointer + AsyncSqliteSaver（SQLite 文件，重启不丢）
- 展示层：MySQL（表 conversations）
- **评估集：249 个测试项（自动生成），覆盖 83 篇文档**
- **随机检索基线：3.0%**（判别空间 54.0 个点）
- 首次 Docker 构建耗时：约 19 分钟（CUDA）→ **5~10 分钟**（CPU）
- 单次请求耗时：2~3 秒（首次含 Reranker 加载 10~20 秒）
- 镜像大小：10.4 GB → **2.42 GB**（content 519 MB）
- Docker 数据位置：`D:\Docker`（55.73 GB）
- C 盘可用空间：10.20 GB → **65.25 GB**
- pip 下载量：3~5 GB（CUDA）→ **~600 MB**（CPU）
- FAISS 索引：已提交 Git（583 chunk 约 1~2 MB）
- checkpoints 持久化：`./data/checkpoints.db`（volume 挂载，容器重建不丢）
- 开发期挂载：`rag_api`、`mcp_test`、`rag_core`、`docs`、`faiss_index`、`data`、`hf_cache`
- `DATA_DIR=/app/data`（从 `/app/rag_api/data` 改过来，避免和 `rag_api` 挂载冲突）
- data/ 目录：`.gitignore` 已忽略
- 公共模块：`rag_core/search.py`（服务端 + 评估脚本共用）
- Redis 服务：`redis:7` 镜像，容器名 `agent-redis`，端口 6379
- 评估可复现：`python eval/eval_retrieval.py` 一条命令跑出所有数字
- 测试集自动生成：`python scripts/generate_testset.py`（成本约 ¥0.2）
- 阈值标定：`python eval/calibrate_distance.py`（正样本 100 + 负样本 25）

## 九、可演示的操作

1. 打开 Swagger UI：http://127.0.0.1:8000/docs
2. 测普通对话：POST /chat（带 Bearer token）
3. 测流式对话：POST /chat/stream（带 Bearer token）
4. 查看对话历史：GET /history?thread_id=user_001（无需鉴权）
5. **演示鉴权**：不带 token → 401；错误 token → 403；正确 token → 200
6. **演示限流**：连打 21 次 → 前 20 次 200，第 21 次 429
7. **演示持久化记忆**：问"MMR 是什么" → **`docker compose down` 完全销毁容器** → `up -d` → 问"它和普通相似度有什么区别" → 仍能理解
8. 演示会话隔离：换 thread_id 问"它是什么"
9. 演示降级策略：问"今天天气怎么样？" → 返回"知识库中没有找到相关信息"
10. 用浏览器打开 client.html，看打字机效果
11. 展示 Docker 一键启动：`docker compose up -d`
12. **展示评估可复现**：`python eval/eval_retrieval.py` → 输出 5 种策略 + 3.0% 随机基线
13. **展示阈值标定**：`python eval/calibrate_distance.py` → 输出正负样本距离分布 + PR 曲线
14. **展示测试集自动生成**：`python scripts/generate_testset.py` → 从 98 篇文档生成 249 条测试项
15. **展示两阶段检索日志**：`docker logs agent-api` → 看到 `[Rerank] 候选 20 条 → Top-3`
16. **展示开发期热更新**：改一行代码 → `docker compose restart agent-api` → 生效，不用 build
17. **展示 Redis 计数器**：`docker exec -it agent-redis redis-cli get "rate:<token>"`
18. 展示 GitHub 仓库：https://github.com/wzy106/rag-agent-service

## 十、RAG 延伸知识点（面试加分）

### 向量库选型对比

| 向量库 | 定位 | 特点 |
|---|---|---|
| InMemoryVectorStore | 内存向量库 | 简单，重启丢数据，适合测试 |
| FAISS | 向量索引库 | 快、轻量，需手动持久化 |
| Chroma | 向量数据库 | 自动持久化、支持元数据过滤 |
| Milvus | 分布式向量库 | 企业级、支持亿级向量 |
| pgvector | PostgreSQL 扩展 | 复用现有 PG |

**FAISS vs Chroma**：
- FAISS 是索引库，只做向量检索，需要手动 save/load
- Chroma 是数据库，自动持久化，支持元数据过滤
- 我选 FAISS 是因为项目规模小、只需向量检索

### Embedding 模型选型

| 模型 | 维度 | 特点 |
|---|---|---|
| bge-small-zh-v1.5 | 512 | 轻量，中文好，我用这个（召回） |
| bge-base-zh-v1.5 | 768 | 效果更好，慢一点 |
| bge-large-zh-v1.5 | 1024 | 效果最好，吃内存 |
| text-embedding-3-small | 1536 | OpenAI 官方，需付费 |

### Rerank 模型选型

| 模型 | 参数量 | 大小 | 特点 |
|---|---|---|---|
| bge-reranker-base | 278M | 1.1 GB | 我用这个，中英双语 |
| bge-reranker-large | 560M | 2.2 GB | 效果更好，慢 2 倍 |
| bge-reranker-v2-m3 | 568M | 2.3 GB | 最新，多语言 |
| Cohere Rerank | 闭源 | API | 需付费，无需部署 |

**bge-reranker-base 的选择理由**：
- 中英双语都支持
- 3 GB 大小，CPU 上单条推理 50~100 ms
- 有开源权重，完全离线

### Rerank 是什么

- **向量检索**：双塔模型，query 和 doc 分别编码，算余弦相似度——**快但粗**
- **Rerank**：交叉编码器，把 `(query, doc)` 拼一起送进模型打分——**慢但精**
- **流程**：向量检索 Top-20 → Rerank → 取 Top-3
- **为什么两阶段**：Reranker 逐条推理太慢，先粗排缩小范围

### RAG 评估指标

| 指标 | 公式 | 含义 |
|---|---|---|
| 召回率 | 命中数 / 总数 | 相关文档有没有被检索到 |
| 精确率 | 平均（相关结果数 / Top-K 总数） | 检索到的有多少是相关的 |
| MRR | 平均（1 / 第一个正确结果的排名） | 第一个正确结果排多前 |
| 多样性 | 平均（Top-K 里不同文档数） | 结果覆盖多少不同来源 |
| 忠实度 | 人工 / RAGAS | 回答是否忠于检索结果 |
| 答案相关性 | 人工 / RAGAS | 回答是否切题 |

工具：RAGAS

### 切分策略

- 固定长度：简单，可能切断语义
- 递归切分：按优先级尝试分隔符（我用这个）
- 语义切分：效果最好但最慢
- 按文档结构切：Markdown 按标题切

### 混合检索

- 向量检索：语义相似，但可能漏关键词
- BM25：关键词精确匹配，但不懂语义
- 混合检索：两者加权合并，取长补短
- RRF 是标准合并算法，公式 `Σ 权重/(k+排名)`

## 十一、MySQL 展示层 + Redis 限流（工程化）

### 为什么还需要 MySQL

| 问题 | 解法 |
|---|---|
| 用户要查看历史对话 | MySQL 展示层，给 /history 接口用 |
| Agent 的记忆 | 交给 LangGraph checkpointer，不经过 MySQL |

**角色划分**：
- **checkpointer（SQLite）**：Agent 的记忆，唯一真相源
- **MySQL**：只做展示层，给 `/history` 接口用
- **Redis**：限流（已落地，`incr + expire`）

### MySQL 对话历史（展示层）

```sql
CREATE TABLE conversations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    thread_id VARCHAR(64),
    role VARCHAR(20),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**数据流**：
```text
用户提问
    ↓
调 Agent（checkpointer 自动加载历史）
    ↓
写 MySQL 展示层（失败不影响主流程）
```

**关键设计**：
- MySQL 写入包 try/except，失败只打日志
- `/history` 接口加 `limit` 上限保护（最大 100）

### Redis 限流（已落地）

**从预留到落地**：原来 Redis 依赖只是"留着备用"，现在正式用于限流。

**方案**：固定窗口，按 API Key 限流，默认 20 次/分钟。

```python
def rate_limit(key: str):
    try:
        r = _get_redis()
        redis_key = f"rate:{key}"
        current = r.incr(redis_key)
        if current == 1:
            r.expire(redis_key, 60)
        if current > RATE_LIMIT_PER_MINUTE:
            raise HTTPException(429, detail="请求过于频繁...")
    except HTTPException:
        raise
    except redis.RedisError as e:
        print(f"[限流] Redis 不可用，放行：{e}")
```

**关键设计**：

| 点 | 说明 |
|---|---|
| 维度 | 按 API Key 限流，复用鉴权拿到的 token |
| 算法 | 固定窗口（incr + expire），简单够用 |
| fail-open | Redis 挂了放行，限流不能挂掉主流程 |
| 配额 | 环境变量 `RATE_LIMIT_PER_MINUTE`，默认 20 |
| `/history` 不限流 | 展示层流量小 |

**为什么保留依赖并落地**：
- 之前删缓存后，Redis 依赖成了"占位符"
- 限流是 Redis 的经典场景，30 行代码就能落地
- "预留做限流" → "已经做了限流"，是两种信誉

### 完整数据流（当前版本）

```text
用户提问（带 thread_id + Bearer token）
    ↓
verify_key 鉴权（错误 → 401/403）
    ↓
rate_limit 限流（超限 → 429；Redis 挂了 → 放行）
    ↓
调 Agent（checkpointer 自动加载历史）
    ↓
写 MySQL 展示层（失败不影响主流程）
    ↓
返回答案
```

### 面试答题要点

| 问题 | 答案 |
|---|---|
| 对话历史怎么持久化？ | LangGraph checkpointer + AsyncSqliteSaver |
| MySQL 用来做什么？ | 展示层，给 /history 接口用 |
| 为什么删掉 Redis 缓存？ | 命中率结构性为 0，是负优化 |
| Redis 现在用来做什么？ | 限流（incr + expire），按 API Key 限流 20/分钟 |
| 限流为什么 fail-open？ | 限流是保护措施，Redis 挂了放行，不能成为新单点 |
| 多实例部署怎么共享状态？ | checkpointer 换 PostgresSaver；限流用 Redis |

### 可优化方向

- Redis 加密码和连接池，生产环境必须
- 用连接池代替每次新建 MySQL 连接（性能优化）
- MySQL 写入改异步（aiomysql）
- 限流算法升级：固定窗口 → 滑动窗口 / 令牌桶（当前规模不需要）

## 十二、FastAPI 服务层设计（面试重点）

### 1. Pydantic 请求/响应模型

```python
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "user_001"

class ChatResponse(BaseModel):
    reply: str
```

**作用**：
- 定义请求体和响应体的结构
- FastAPI 自动做类型校验，字段错/类型错返回 422
- 自动生成 Swagger UI 文档
- IDE 有自动补全

### 2. lifespan 生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    ...
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(data_dir / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        agent = create_agent(model, tools=tools, checkpointer=saver, ...)
        ensure_table()
        yield

app = FastAPI(lifespan=lifespan)
```

**为什么不用模块顶层？**
- 顶层代码在 import 时就执行，FastAPI 事件循环还没启动
- 在顶层调 `asyncio.run()` 会和 uvicorn 事件循环冲突
- lifespan 在事件循环里执行，可以 `await`

**为什么用 `async with`**：`AsyncSqliteSaver` 需要在生命周期内保持连接。`async with` 保证服务停止时正确关闭。

### 3. MySQL 幂等建表

```python
def ensure_table():
    conn = pymysql.connect(...)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (...)
            """)
        conn.commit()
    finally:
        conn.close()
```

**关键点**：
- `IF NOT EXISTS`：表存在则跳过，**幂等**
- 在 lifespan 启动时调用，**服务自愈**
- `try/finally` 保证连接一定关闭

### 4. SSE 流式接口的中间步骤过滤

**问题**：Agent 在调用工具前会有英文思考，不该给用户看。

**解法**：用 `saw_tool` 标志区分。

```python
saw_tool = False
pending = []
started = False

async for chunk in agent.astream(...):
    msg_chunk, metadata = chunk
    if msg_chunk.type == "tool":
        saw_tool = True
        pending = []
        continue
    if not msg_chunk.content:
        continue
    if saw_tool:
        yield f"data: {json.dumps({'text': msg_chunk.content})}\n\n"
        started = True
    else:
        pending.append(msg_chunk.content)

# 兜底：模型不输出文本时给友好提示
if not started and not pending:
    fallback = "抱歉，我无法生成回答。"
    yield f"data: {json.dumps({'text': fallback})}\n\n"
```

**已知不足**：用三个布尔变量表达状态，边界情况可能出错。生产环境应该用 `metadata["langgraph_node"]` 做来源判断 + 显式状态机。

### 5. SSE 格式规范

```python
yield f"data: {json.dumps({'text': msg_chunk.content}, ensure_ascii=False)}\n\n"
```

| 部分 | 作用 |
|---|---|
| `data: ` | SSE 协议要求的前缀 |
| `json.dumps(...)` | 转成 JSON |
| `ensure_ascii=False` | 保留中文 |
| `\n\n` | 两条换行表示结束 |

**结束标记**：`yield "data: [DONE]\n\n"`

**重要设计**：`[DONE]` **不能放在 `finally`**。客户端断连时会调 `aclose()`，在 `finally` 里 `yield` 会抛 `RuntimeError: async generator ignored GeneratorExit`。

**正确做法**：
```python
try:
    ...
    yield "data: [DONE]\n\n"      # 成功路径
except asyncio.CancelledError:
    logger.info("客户端断连")       # 不要 yield
    raise
except Exception as e:
    yield f"data: {json.dumps({'error': friendly_error(e)})}\n\n"
    yield "data: [DONE]\n\n"      # 失败路径
finally:
    # 只做清理，不 yield
    print(f"[请求] 耗时 ...")
```

### 6. 异常处理（友好错误提示）

```python
def friendly_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    if status == 401:
        return "模型服务认证失败，请联系管理员"
    if status == 429:
        return "请求过于频繁，请稍后再试"
    if isinstance(e, asyncio.TimeoutError):
        return "请求超时，请稍后再试"
    if isinstance(e, pymysql.MySQLError):
        return "存储服务暂时不可用，请稍后再试"
    return "服务暂时不可用，请稍后再试"
```

**为什么不子串匹配**：`"rate" in msg` 会误匹配 `generate`、`operate`。用异常类型 + HTTP 状态码更准确。

**服务端**：完整异常打日志，方便排查
**客户端**：只看到友好提示，不泄漏内部信息

### 7. MySQL 连接参数用环境变量

```python
conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", "root123"),
    database=os.getenv("MYSQL_DATABASE", "agent"),
    charset="utf8mb4"
)
```

**为什么**：本地开发用 `localhost`，Docker 容器里要用服务名 `mysql`。写死了容器里连不上。

### 8. Agent 护栏

**问题**：Agent 可能无限调用工具，或者单次输出过长。

**方案**：两道护栏。

```python
RECURSION_LIMIT = 10
MAX_TOKENS = 2000

model = ChatOpenAI(..., max_tokens=MAX_TOKENS)
config = {"recursion_limit": RECURSION_LIMIT}
```

### 9. 持久化记忆（AsyncSqliteSaver）

**问题**：`InMemorySaver` 是进程内存，服务重启后记忆丢失。手动从 MySQL 加载历史又和 checkpointer 打架。

**方案**：用 `AsyncSqliteSaver` 做持久化 checkpointer，路径由 `DATA_DIR` 决定。

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
data_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = str(data_dir / "checkpoints.db")
async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
    agent = create_agent(
        model, tools=tools,
        checkpointer=saver,
        system_prompt="..."
    )
```

**关键设计**：

| 角色 | 之前 | 之后 |
|---|---|---|
| 记忆源 | InMemorySaver + MySQL 手动拼 | AsyncSqliteSaver（唯一真相源） |
| MySQL | 记忆源 | 降级为展示层（给 /history 用） |
| 消息格式 | OpenAI dict 手动拼 | 统一交给 LangGraph |
| DB 路径 | 硬编码在源码目录 | `DATA_DIR` 环境变量，本地/容器两套值 |
| 容器重建 | DB 丢失 | volume 挂载，不丢 |

### 10. API Key 鉴权

**问题**：接口完全裸奔，任何人都能调。

**方案**：用 `Header` + `Depends` 做 Bearer token 校验。

```python
API_KEY = os.getenv("API_KEY", "dev-key-change-me")

async def verify_key(authorization: str = Header(default="")) -> str:
    if not authorization:
        raise HTTPException(401, detail="缺少 Authorization 头")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Authorization 格式错误")
    token = authorization[7:].strip()
    if token != API_KEY:
        raise HTTPException(403, detail="API Key 无效")
    return token

@app.post("/chat")
async def chat(payload: ChatRequest, token: str = Depends(verify_key)):
    ...
```

**关键设计**：

| 点 | 说明 |
|---|---|
| `Header` | FastAPI 自动从请求头取值 |
| `Depends` | 依赖注入，接口调用前先跑 `verify_key` |
| `401` vs `403` | 401 = 没带凭证；403 = 带了但不对 |
| `API_KEY` 环境变量 | 密码外置，代码里不出现真实密钥 |
| `${API_KEY:-默认}` | compose 从宿主机读，注入容器 |

**验证**：

```cmd
:: 不带 token → 401
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"message":"MMR 是什么？"}'
{"detail":"缺少 Authorization 头"}

:: 错误 token → 403
curl ... -H "Authorization: Bearer wrong" ...
{"detail":"API Key 无效"}

:: 正确 token → 200
curl ... -H "Authorization: Bearer your-secret-key-here" ...
{"reply":"..."}
```

**面试能讲**：从"裸奔"到"有鉴权"，是生产化的第一步。用 `Depends` 让鉴权逻辑和业务逻辑解耦，`/chat` 和 `/chat/stream` 共用同一个 `verify_key`。`API_KEY` 从环境变量读，本地 `.env` 一份、生产服务器一份，代码里不出现真实密钥。

### 10.5 Redis 限流

**问题**：加了鉴权只防"谁能调"，不防"调多少次"。一个 Key 可以无限打。

**方案**：固定窗口限流，按 API Key 计，20 次/分钟。

```python
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

def rate_limit(key: str):
    try:
        r = _get_redis()
        redis_key = f"rate:{key}"
        current = r.incr(redis_key)
        if current == 1:
            r.expire(redis_key, 60)
        if current > RATE_LIMIT_PER_MINUTE:
            raise HTTPException(429, detail="请求过于频繁...")
    except HTTPException:
        raise
    except redis.RedisError as e:
        print(f"[限流] Redis 不可用，放行：{e}")
```

**接口挂上**：

```python
@app.post("/chat")
async def chat(payload: ChatRequest, token: str = Depends(verify_key)):
    rate_limit(token)
    ...
```

`verify_key` 从 `-> None` 改成 `-> str`，返回 token 供限流用。

**关键设计**：

| 点 | 说明 |
|---|---|
| 维度 | 按 API Key 限流，复用鉴权拿到的 token |
| 算法 | 固定窗口（`incr + expire`），不做滑动窗口 |
| fail-open | Redis 挂了放行，限流不能成为新单点 |
| 配额 | 环境变量 `RATE_LIMIT_PER_MINUTE`，默认 20 |
| `/history` 不限流 | 展示层流量小 |

**验证**：

```cmd
for /L %i in (1,1,21) do @curl.exe ... -w "%i : %{http_code}\n" ...
:: 前 20 次 200，第 21 次 429
```

**面试话术**：从"预留依赖"到"真实使用"，Redis 的角色从"负优化"（之前删掉的缓存）变成"限流"。用 `incr + expire` 做固定窗口，比滑动窗口简单，够用。**Redis 挂了要放行**——限流是保护措施，不能让它成为新的单点故障。鉴权 + 限流是生产环境的两层保护：鉴权防"谁能调"，限流防"调多少次"。

### 11. 面试高频问题

| 问题 | 答案要点 |
|---|---|
| Pydantic 在 FastAPI 里做什么？ | 请求/响应模型、类型校验、自动文档 |
| lifespan 是什么？ | 启动和关闭钩子，替代废弃的 on_event |
| 为什么用 lifespan 不用顶层代码？ | 顶层会和 uvicorn 事件循环冲突 |
| 流式接口怎么过滤中间步骤？ | `saw_tool` 标志分界，`pending` 缓存工具前内容 |
| SSE 格式是什么？ | `data: 内容\n\n`，结束发 `[DONE]` |
| 为什么 [DONE] 不能放 finally？ | 客户端断连时 aclose() 抛 GeneratorExit，finally 里 yield 会抛 RuntimeError |
| Agent 死循环怎么防？ | `recursion_limit=10` + `max_tokens=2000` |
| Agent 多轮对话怎么实现？ | LangGraph checkpointer + AsyncSqliteSaver + DATA_DIR |
| 容器重建后记忆怎么不丢？ | checkpoints volume 挂载到宿主机 ./data/ |
| 错误信息怎么处理？ | `friendly_error` 映射，用异常类型判断 |
| 检索不到内容怎么处理？ | 距离阈值降级（标定 0.96），返回"没找到"，不调 LLM |
| 接口鉴权怎么做？ | Bearer token + `Depends` 依赖注入；`API_KEY` 从环境变量读 |
| 限流怎么做？ | Redis 固定窗口 `incr + expire`，按 API Key，fail-open |
| 为什么用 `Depends` 不用手动调？ | 鉴权逻辑抽出来，多接口共用，改一处生效 |
| 开发期怎么快速迭代？ | volume 挂载代码/语料，`docker compose restart` 不用 rebuild |
| RAG 评估用什么指标？ | 召回率 + 精确率 + MRR + 多样性 + 随机基线 |
| 评估怎么保证可复现？ | 抽公共 `rag_core/search.py`，一条命令跑出所有数字 |
| 阈值怎么定？ | 正负样本距离分布 + PR 曲线标定（100 正 + 25 负） |
| 为什么要 Rerank？ | 双塔召回粗，交叉编码器精排 |

## 十三、RAG 评估（面试重点）

### 评估方法

构建 **249 条测试项**（覆盖 83 篇文档），测试集用脚本自动生成：
1. 遍历所有 chunk，按 source 分组
2. 每篇文档挑 2 个最长 chunk
3. 调 DeepSeek API 生成 3 个问题
4. 标注 `expected_sources = 该文档的 source`

**关键设计**：
- 问题口语化、角度多样、能被对应文档回答
- 13 篇极短文档（chunk < 2）被跳过
- 平均每篇 3 题

### 测试集自动生成脚本

`scripts/generate_testset.py`：

```python
async def generate_questions(source: str, chunks: list, sem: asyncio.Semaphore):
    async with sem:
        # 挑两个最长 chunk
        sorted_chunks = sorted(chunks, key=lambda c: len(c.page_content), reverse=True)
        picked = sorted_chunks[:CHUNKS_PER_DOC]
        if len(picked) < 2:
            return None

        prompt = PROMPT_TEMPLATE.format(
            chunk1=picked[0].page_content[:800],
            chunk2=picked[1].page_content[:800],
        )
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        content = resp.choices[0].message.content.strip()
        # 去掉可能的 ```json ``` 包裹
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        questions = json.loads(content)
        return {"source": source, "questions": questions[:QUESTIONS_PER_DOC]}
```

**成本**：约 ¥0.2（98 篇 × 1 次 API 调用）
**时间**：并发 5，约 1~2 分钟

### 评估脚本核心

```python
def evaluate_full(search_fn, name, k=3):
    hit = 0
    precision_sum = 0
    mrr_sum = 0
    diversity_sum = 0

    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]
        unique_sources = set(sources)

        # 召回率
        if any(exp in sources for exp in item["expected_sources"]):
            hit += 1

        # 精确率
        if sources:
            relevant = sum(1 for s in sources if s in item["expected_sources"])
            precision_sum += relevant / len(sources)

        # MRR
        for rank, s in enumerate(sources, 1):
            if s in item["expected_sources"]:
                mrr_sum += 1 / rank
                break

        # 多样性
        diversity_sum += len(unique_sources)

    n = len(test_set)
    return {
        "recall": hit / n,
        "precision": precision_sum / n,
        "mrr": mrr_sum / n,
        "diversity": diversity_sum / n,
    }
```

### 随机基线（组合公式精确计算）

```python
from math import comb
from collections import Counter

def random_baseline(test_set, all_docs, k=3):
    """
    从全部 chunk 里随机抽 k 个，算 source 级召回率期望。
    P(至少命中一个期望 source) = 1 - C(N - n_expected, k) / C(N, k)
    """
    n_total = len(all_docs)
    source_counts = Counter(doc.metadata["source"] for doc in all_docs)

    total = 0.0
    for item in test_set:
        n_expected = sum(source_counts.get(s, 0) for s in item["expected_sources"])
        prob = 1 - comb(n_total - n_expected, k) / comb(n_total, k)
        total += prob
    return total / len(test_set)
```

### 评估结果（98 篇文档 / 583 chunk / 249 测试项）

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 50.6% | 0.23 | 0.39 | 2.63 |
| MMR | 37.8% | 0.15 | 0.33 | 2.79 |
| MMR + 按来源去重 | 40.6% | 0.14 | 0.34 | **3.00** |
| 混合检索（带降级 0.96） | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |
| 随机基线 | **3.0%** | — | — | — |

> **可复现**：`python eval/eval_retrieval.py` 一条命令跑出上面所有数字。评估脚本从 `rag_core.search` import 检索函数，和服务端部署的是同一份代码。

### 关键发现

**1. 随机基线从 84.1% 降到 3.0%——判别空间 54.0 个点**

- 旧语料：2 篇文档 / 102 chunk / 30 测试项 → 随机基线 84.1%，判别空间 15.9 个点
- 新语料：98 篇文档 / 583 chunk / 249 测试项 → 随机基线 3.0%，判别空间 **54.0 个点**
- **旧数字全 100% 是"天花板效应"**——测试集太简单，看不出策略差异

**2. Rerank 是精确率的最后一块拼图**

- 混合检索召回率 52.6% → Rerank 后 **57.0%**（+4.4%）
- MRR 0.42 → **0.47**（+0.05）
- **主要提升召回率**——Rerank 把 Top-20 里被埋没的相关 chunk 顶上来

**3. 混合检索优于纯相似度**

- 52.6% vs 50.6%（**+2.0%**）
- 说明 BM25 补的关键词匹配确实有用

**4. MMR 在"精确匹配"任务上吃亏**

- 召回率只有 37.8%（低于相似度 50.6%）
- 但多样性较高（2.79）
- **多样性导向的代价**——选多角度的结果，但可能漏掉最相关的

**5. 精确率和多样性呈明显 trade-off**

| 策略 | 精确率 | 多样性 |
|---|---|---|
| 混合检索 + Rerank | 0.25 | 2.56 |
| MMR + 去重 | 0.14 | **3.00** |

**6. 选哪个策略取决于业务**

| 场景 | 推荐策略 |
|---|---|
| 追求精确（如事实查询） | **混合检索 + Rerank** |
| 需要多角度（如分析任务） | MMR + 去重 |
| 平衡 | MMR |

### 失败案例分析

**案例 1**：口语化表达
- 用户问："怎么让回答一个字一个字显示出来？"
- 期望文档：`readme/04_启动.md`
- 实际命中：`notes_interview/q05_流式输出怎么实现的.md`
- 原因：口语化表达和文档术语有语义鸿沟

**案例 2**：跨主题问题
- 用户问："多 Agent 系统怎么保证消息不丢？"
- 期望文档：`stories/18_故事_17双源记忆打架架构重构.md`
- 实际命中：`notes_interview/q12_Agent_多轮对话怎么实现.md`
- 原因：语义相近但主题不同

### 已知不足

**评估层面**：
- **文档级召回而非 chunk 级**：用 `source`（相对路径）判断命中，同一文档的多个 chunk 算命中。粒度还可以更细
- **source 级评估可能高估效果**：估计高估 8~14 个百分点；后续需要 chunk 级标注或人工复核
- **chunk 级标注成本高**：需要人工标注每个问题的"正确 chunk id"，耗时大
- **没有端到端指标**：检索对了不等于回答对了。应该加 LLM-as-judge 或人工标注的 answer correctness

**架构层面**：
| 位置 | 问题 | 生产级解法 |
|---|---|---|
| `/chat` 和 `/chat/stream` | pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环 | 用 `aiomysql` |
| `save_conversation` | 每次请求新建连接，没有连接池 | 用连接池（如 `DBUtils`） |
| 流式状态机 | 用三个布尔变量表达状态，边界情况可能出错 | 改用 `metadata["langgraph_node"]` + 显式状态机 |
| MCP Server 用了私有 API | `vectorstore.docstore._dict` 是内部 API，版本升级可能坏 | 构建索引时自己维护 chunks 列表并 pickle 落盘 |
| **Reranker 性能** | CPU 单次 50~100 ms，249 条需 4~8 分钟 | 更小模型 / GPU / 缓存 |
| **chunk 级评估** | 现在是文档级 | 标注 chunk id |

### 改进方案

| 方案 | 原理 | 状态 |
|---|---|---|
| 混合检索 | 向量 + BM25，加权合并 | ✅ 已做 |
| 降级策略 | 距离阈值拦截无关问题 | ✅ 已做 |
| 多指标评估 | 召回率 + 精确率 + MRR + 多样性 | ✅ 已做 |
| 随机基线 | 组合公式精确计算 | ✅ 已做 |
| 友好错误提示 | friendly_error 映射 | ✅ 已做 |
| 删掉无效缓存 | 命中率结构性为 0 | ✅ 已做 |
| Docker 多阶段构建 | 3.51 GB → 2.42 GB | ✅ 已做 |
| CPU 版 torch | 下载量 5 GB → 600 MB | ✅ 已做 |
| 模型缓存 volume 挂载 | 重建不重下 | ✅ 已做 |
| checkpoints volume 挂载 | 容器重建不丢记忆 | ✅ 已做 |
| 开发期 volume 挂载 | 改代码不用 rebuild | ✅ 已做 |
| HF_HOME 三级探测 | 本地/容器都能跑 | ✅ 已做 |
| 抽公共检索模块 | rag_core/search.py | ✅ 已做 |
| 阈值标定 | 正负样本距离分布 + PR 曲线 | ✅ 已做（MAX_DISTANCE 1.0→0.96） |
| Rerank 精排 | bge-reranker-base 对 Top-20 重排 | ✅ 已做（召回率 +4.4%） |
| API Key 鉴权 | Bearer token + Depends | ✅ 已做 |
| Redis 限流 | 按 API Key，20/分钟，fail-open | ✅ 已做 |
| **语料扩充** | 2 篇 → 98 篇 | ✅ 已做 |
| **测试集自动生成** | 30 条 → 249 条 | ✅ 已做 |
| Chunk 级评估 | 标注 chunk id，不只标文档 | ⏳ 待做 |
| 端到端评估 | LLM-as-judge 或人工标注 | ⏳ 待做 |

### 面试答题模板

> "我构建了 249 个测试项的 RAG 评估集，故意让问题不含文档关键词。测试集是通过脚本自动生成的——遍历 98 篇文档的 chunk，每篇挑 2 个最长 chunk，调 DeepSeek API 生成 3 个问题。用了四个指标 + 随机基线：召回率、精确率、MRR、多样性。随机基线用组合公式精确算，从 583 个 chunk 里随机抽 3 个，命中期望 source 的概率是 3.0%。检索 pipeline 是两阶段：混合检索召回 Top-20（向量 + BM25 + 加权 RRF），再用 bge-reranker-base 交叉编码器精排 Top-3。Rerank 把召回率从 52.6% 提到 57.0%，MRR 从 0.42 提到 0.47。降级阈值 MAX_DISTANCE 也是标定出来的——正样本 100 条 + 负样本 25 条，扫 101 个候选点，最终取 0.96，P=0.990、R=1.000、F1=0.995。所有数字都能用 `python eval/eval_retrieval.py` 复现。**这里有个真实的迭代故事**：我第一版语料只有 2 篇文档、测试集 30 条，随机基线 84.1%——天花板效应让 Rerank 提升看起来只有 +0.04 精确率。**扩语料到 98 篇、测试集到 249 条后**，随机基线降到 3.0%，Rerank 的真实增益才显现出来，召回率 +4.4%、MRR +0.05。"

## 十四、Docker 部署完整方案（面试重点）

### 14.1 最终 Dockerfile

```dockerfile
# ---------- Stage 1: 构建依赖 ----------
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 阿里云源 + PyTorch CPU 源
RUN pip install --no-cache-dir --prefix=/install \
        -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --trusted-host mirrors.aliyun.com \
        --trusted-host download.pytorch.org \
        --timeout 300 \
        --retries 10

# ---------- Stage 2: 运行时 ----------
FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local

ENV HF_HOME=/root/.cache/huggingface
ENV HF_ENDPOINT=https://hf-mirror.com

COPY docs/ ./docs/
COPY rag_core/ ./rag_core/
COPY faiss_index/ ./faiss_index/
COPY mcp_test/mcp_rag_server.py ./mcp_test/mcp_rag_server.py
COPY rag_api/api_agent.py ./rag_api/api_agent.py

EXPOSE 8000
CMD ["uvicorn", "rag_api.api_agent:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 14.2 最终 docker-compose.yml

```yaml
services:
  mysql:
    image: mysql:8
    container_name: agent-mysql
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: root123
      MYSQL_DATABASE: agent
    volumes:
      - mysql_data:/var/lib/mysql
    networks:
      - agent-net

  redis:
    image: redis:7
    container_name: agent-redis
    ports:
      - "6379:6379"
    networks:
      - agent-net
    restart: unless-stopped

  agent-api:
    build: .
    container_name: agent-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_USER=root
      - MYSQL_DATABASE=agent
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - HF_ENDPOINT=https://hf-mirror.com
      - HF_HOME=/root/.cache/huggingface
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
      - DATA_DIR=/app/data
      - API_KEY=${API_KEY:-dev-key-change-me}
      - RATE_LIMIT_PER_MINUTE=${RATE_LIMIT_PER_MINUTE:-20}
    volumes:
      - ./hf_cache:/root/.cache/huggingface
      - ./data:/app/data
      # === 开发期挂载：改代码不用 rebuild ===
      - ./rag_api:/app/rag_api
      - ./mcp_test:/app/mcp_test
      - ./rag_core:/app/rag_core
      - ./docs:/app/docs
      - ./faiss_index:/app/faiss_index
    depends_on:
      - mysql
      - redis
    networks:
      - agent-net
    restart: unless-stopped

volumes:
  mysql_data:

networks:
  agent-net:
    driver: bridge
```

### 14.3 requirements.txt 关键两行

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

**原理**：PyTorch 官方 CPU 源里 torch 版本是 `2.9.0+cpu`，按 PEP 440 规则 `2.9.0+cpu > 2.9.0`，pip 自动选 CPU 版。

### 14.4 部署踩坑时间线

**2026-10-08（Docker 部署问题闭环）**

| 时间 | 问题 | 动作 |
|---|---|---|
| 19:00 | `unpigz corrupted`，构建失败 | 查 C 盘 → 0 字节 |
| 19:10 | 清理 C 盘 | Free 10 GB |
| 19:20 | 项目搬到 D:\Projects\Agent | robocopy 绕过文件锁 |
| 19:30 | Docker 数据迁移 | Settings → `D:\Docker` → C 盘 65 GB |
| 19:50 | 重建 Dockerfile | 构建成功，2.42 GB |
| 20:00 | 启动容器 | ✅ |
| 20:10 | 首次 `/chat` 报 `FileNotFoundError` | MCP Server 找不到模型缓存 |
| 20:20 | 宿主机 `snapshot_download` | 192 MB 下完 |
| 20:30 | 重启容器 | ✅ |
| 21:00 | 发现 `docker compose down` 后记忆全丢 | checkpoints 无 volume |
| 21:10 | 改 `DATA_DIR` + compose 加 volume | ✅ |

**2026-10-09**：抽公共模块 + 加随机基线

**2026-10-10**：正负样本标定 MAX_DISTANCE 1.1→1.0

**2026-10-11**：Rerank 精排 + 语料 2→96 篇 + 测试集 30→249 条

**2026-10-12**：阈值重标定 MAX_DISTANCE 1.0→0.96；notes/07 拆成 25 个独立面试题文件，语料同步为 98 篇 / 583 chunk

**2026-10-13**：加 API Key 鉴权 + 开发期 volume 挂载

**2026-10-14**：加 Redis 限流（`incr + expire`，fail-open）

### 14.5 面试话术（Docker 部署）

> "我的项目用 Docker 多阶段构建 + CPU 版 torch，镜像从 3.51 GB 优化到 2.42 GB。Docker 数据落在 D 盘避免占满 C 盘。MCP Server 需要两个模型：bge-small-zh-v1.5（embedding，192 MB）和 bge-reranker-base（rerank，3.36 GB），我都下到宿主机 `hf_cache/` 然后 volume 挂载进容器。checkpoints.db 也挂到宿主机 `./data/`，这样 `docker compose down` 后容器重建，Agent 的记忆也不会丢。检索逻辑抽到 `rag_core/search.py`，服务端和评估脚本共用同一份代码。降级阈值 MAX_DISTANCE 通过正负样本距离分布标定为 0.96，P=0.990、R=1.000、F1=0.995。检索用两阶段 pipeline：混合检索召回 Top-20 + Rerank 精排 Top-3，召回率从 52.6% 提到 57.0%。当前知识库 98 篇文档 / 583 chunk，评估集 249 条，随机基线 3.0%。接口加了 API Key 鉴权和 Redis 限流——鉴权防'谁能调'，限流防'调多少次'，限流用 `incr + expire` 固定窗口，fail-open。开发期把代码和语料目录做成 volume 挂载，改代码只要 `docker compose restart`，不用 rebuild。docker compose up -d 一条命令就能起完整服务：MySQL + Redis + Agent API。"

## 十五、项目新增数据（更新）

### 部署相关（2026-10-08 ~ 2026-10-14）

- Docker 数据位置：`D:\Docker`（55.73 GB）
- C 盘可用空间：10.20 GB → **65.25 GB**
- 镜像大小：3.51 GB → **2.42 GB**（content 519 MB）
- 构建耗时：33 分钟（CUDA）→ **5~10 分钟**（CPU）
- pip 下载量：3~5 GB → **~600 MB**
- checkpoints 持久化：`./data/checkpoints.db`（volume 挂载）
- 开发期挂载：`rag_api`、`mcp_test`、`rag_core`、`docs`、`faiss_index`
- 公共模块：`rag_core/search.py`
- 阈值标定：`eval/calibrate_distance.py`（正样本 100 + 负样本 25）
- MAX_DISTANCE：1.1 → 1.0 → **0.96**
- 阈值标定结果：P=0.990、R=1.000、F1=0.995
- 幻觉率：9.1% → **约 1.0%**（漏答 0%）
- Reranker 模型：bge-reranker-base（3.36 GB，本地缓存）
- **Rerank 提升：召回率 52.6% → 57.0%，MRR 0.42 → 0.47**
- **语料扩充：2 篇 → 98 篇，chunk 102 → 583**
- **测试集扩充：30 条 → 249 条，随机基线 84.1% → 3.0%**
- **判别空间：15.9 个点 → 54.0 个点**
- **API Key 鉴权**：Bearer token + `Depends` 依赖注入
- **Redis 限流**：按 API Key，20 次/分钟，`incr + expire`，fail-open
- **开发期 volume 挂载**：改代码不用 rebuild

### Git 仓库状态（最新）

- 新增 commit：
  - `59e67a3` feat: add API Key auth + dev volume mounts
  - `72e038c` docs: sync notes, docs corpus, faiss index after threshold recalibration
  - `a4a7757` feat: 阈值重标定 + notes/07 拆分后语料同步（98 篇 / 583 chunk，MAX_DISTANCE 0.96）
  - `2eb5ccb` feat: 扩测试集 30→249 条（覆盖 83 篇文档），拆分面试题为 25 文件，随机基线 84.1%→3.5%
  - `5fe1817` feat: 扩语料 2 篇 → 72 篇（拆分三个笔记 + LangChain 笔记），chunk 102 → 547
  - `7feaade` docs: 补充 Rerank 精排的说明、评估数据和踩坑记录
  - `37d47f1` feat: 加 Rerank 精排（bge-reranker-base），精确率 0.86 → 0.90
  - `f89cda3` docs: 补充 MAX_DISTANCE 阈值标定（1.1 → 1.0）的说明和实验数据
  - `466e671` feat: 标定 MAX_DISTANCE 阈值（1.1 → 1.0），幻觉率从 9.1% 降到 3.3%
  - `99af642` refactor: 抽公共 hybrid_search 到 rag_core，评估脚本加随机基线
  - `6f9f286` fix: 修复 HF 缓存路径和 checkpoints 持久化
- GitHub：https://github.com/wzy106/rag-agent-service

## 十六、项目数据汇总（最新）

- MySQL 表名：`conversations`（展示层）
- 新增接口：`GET /history?thread_id=user_001&limit=20`
- **知识库：98 篇文档 / 583 chunk**
- 子目录文件数：langchain 12、notes 15、notes_interview 25、readme 13、stories 33
- **评估集：249 个测试项（覆盖 83 篇文档，脚本自动生成）**
- **随机检索基线：3.0%**（判别空间 54.0 个点）
- 相似度：召回率 50.6%，精确率 0.23，MRR 0.39，多样性 2.63
- MMR：召回率 37.8%，精确率 0.15，MRR 0.33，多样性 2.79
- MMR + 去重：召回率 40.6%，精确率 0.14，MRR 0.34，多样性 **3.00**
- 混合检索：召回率 52.6%，精确率 0.24，MRR 0.42，多样性 2.57
- **混合检索 + Rerank：召回率 57.0%，精确率 0.25，MRR 0.47，多样性 2.56**
- 混合检索参数：rrf_k=60, vector_weight=0.7, bm25_weight=0.3
- Rerank 参数：recall_k=20, final_k=3, RERANKER_MODEL="BAAI/bge-reranker-base"
- 降级阈值：MAX_DISTANCE=0.96（标定得出，P=0.990, R=1.000, F1=0.995）
- 阈值标定：正样本 100 条（test_set_v2.json，seed=42）+ 负样本 25 条
- 正样本距离：0.363~0.958，中位数 0.709
- 负样本距离：0.813~1.420，中位数 1.176
- Rerank 增益：召回率 +4.4%，MRR +0.05
- 混合检索 vs 相似度：+2.0%
- 护栏参数：recursion_limit=10, max_tokens=2000
- 鉴权：Bearer token（`/chat` 和 `/chat/stream`），`API_KEY` 从环境变量读
- 限流：Redis 固定窗口（`incr + expire`），按 API Key 限流 20 次/分钟，fail-open
- Redis 服务：`redis:7` 镜像，容器名 `agent-redis`，端口 6379
- 记忆：LangGraph checkpointer + AsyncSqliteSaver（容器重建不丢）
- 公共模块：`rag_core/search.py`
- 开发期挂载：`rag_api`、`mcp_test`、`rag_core`、`docs`、`faiss_index`、`data`、`hf_cache`
- `DATA_DIR=/app/data`（从 `/app/rag_api/data` 改过来，避免和 `rag_api` 挂载冲突）
- 评估可复现：`python eval/eval_retrieval.py`
- 测试集生成：`python scripts/generate_testset.py`（成本约 ¥0.2）
- 阈值标定：`python eval/calibrate_distance.py`
- GitHub 仓库：https://github.com/wzy106/rag-agent-service

---

**备注**：如果新对话需要，可以直接复制本笔记。当前项目已**完整跑通、评估可复现、阈值有实验支撑、Rerank 精排就位、API Key 鉴权、Redis 限流、开发期 volume 挂载、语料 98 篇 / 583 chunk、测试集 249 条**，可进入面试冲刺阶段。