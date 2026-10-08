# RAG Agent 项目复盘笔记

> 用途：面试前自用，把项目讲清楚、把问题想明白。
> 面试话术见 `LESSONS_LEARNED.md`

## 一、项目简介（30 秒版本）

基于 LangGraph + MCP + RAG 的企业级 Agent 服务，用 FastAPI 提供 HTTP 接口，支持 SSE 流式输出，Docker 容器化部署。

## 二、架构图

```text
客户端（浏览器 / curl / 前端）
    ↓ HTTP POST /chat/stream
FastAPI（api_agent.py）
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
    │  降级判断（距离阈值）              │
    │       ↓ 通过                      │
    │  混合检索：                        │
    │  向量检索 Top-9  +  BM25 Top-9    │
    │       ↓                           │
    │  加权 RRF 合并（向量0.7+BM25 0.3） │
    │       ↓                           │
    │  Top-3 返回                        │
    │                                    │
    │  降级：距离超阈值 → 返回"没找到"    │
    └─────────────────────────────────┘
                   ↓
    检索结果返回给 Agent（循环第 ④ 步）
                   ↓ SSE 流式返回
客户端

记忆层：
- LangGraph checkpointer 按 thread_id 管状态（唯一真相源）
- AsyncSqliteSaver 持久化到 SQLite 文件（重启不丢）
- MySQL 只做展示层（给 /history 接口用）

缓存层：
- 暂时没有缓存。历史版本用过 Redis，但命中率结构性为 0，已删除。
- Redis 依赖保留在 requirements.txt 和 docker-compose.db.yml 中，
  下一步计划用于限流（incr + expire）。
```

## 三、技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | LangChain 1.4 + LangGraph 1.2 |
| 模型 | DeepSeek（OpenAI 兼容协议） |
| 工具协议 | MCP（FastMCP 3.4.7） |
| 检索 | sentence-transformers + FAISS + BM25 + 加权 RRF |
| 降级 | 向量距离阈值（避免幻觉） |
| 记忆持久化 | AsyncSqliteSaver（生产可换 PostgresSaver） |
| 展示层 | MySQL（对话历史，只读） |
| 服务 | FastAPI + Uvicorn |
| 部署 | Docker + Docker Compose |
| 流式 | SSE（Server-Sent Events） |

## 四、核心功能

- **持久化记忆**：LangGraph checkpointer + AsyncSqliteSaver，重启不丢
- RAG 检索增强（本地 embedding + FAISS 向量库）
- 混合检索（向量 + BM25 + 加权 RRF）
- 降级策略（距离阈值，避免幻觉）
- MCP 工具解耦（Agent 和工具分进程）
- SSE 流式输出（打字机效果）
- 中间步骤过滤（只输出最终回答）
- **Agent 护栏**（recursion_limit=10，max_tokens=2000）
- **友好异常处理**（异常类型映射，不泄漏内部信息）
- 请求日志（收到消息 + 耗时统计）
- FAISS 索引持久化（避免重复 embedding）
- MySQL 持久化对话历史（展示层）
- RAG 评估（召回率 + 精确率 + MRR + 多样性）

## 五、RAG 完整链路（面试重点）

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
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()
```

- 用本地 `BAAI/bge-small-zh-v1.5` 模型，512 维
- 封装成 LangChain 的 `Embeddings` 接口
- 完全离线，不消耗 API

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
- 索引文件加进 `.gitignore`

### 5. 检索策略

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
def hybrid_search(query, k=3, rrf_k=60, vector_weight=0.7, bm25_weight=0.3):
    # 向量检索 Top-9
    vector_results = vectorstore.similarity_search(query, k=k*3)
    vector_ranks = {doc.page_content: i for i, doc in enumerate(vector_results)}

    # BM25 检索 Top-9
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

#### 5.4 降级策略（避免幻觉）

**问题**：检索不到相关内容时，如果直接调 LLM，模型可能基于无关片段产生幻觉。

**方案**：在检索层加距离阈值，超过就返回"没找到"。

```python
MAX_DISTANCE = 1.1   # 向量 L2 距离阈值

def hybrid_search(query, k=3, max_distance=MAX_DISTANCE, ...):
    vector_results_with_scores = vectorstore.similarity_search_with_score(query, k=k*3)

    if not vector_results_with_scores:
        return []

    # 降级判断：最佳向量距离太大 → 认为知识库中无相关内容
    best_distance = vector_results_with_scores[0][1]
    if best_distance > max_distance:
        return []

    # 正常逻辑
    ...
```

`search_knowledge` 工具处理空结果：

```python
@mcp.tool
def search_knowledge(query: str) -> str:
    results = hybrid_search(query, k=3)
    if not results:
        return "知识库中没有找到相关信息。"
    return "\n\n".join([f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in results])
```

**阈值怎么定（实测数据）**：

| 类别 | 问题 | 距离 |
|---|---|---|
| 正常 | MMR 是什么？ | 0.82 |
| 正常 | RAG 完整流程 | 0.89 |
| 正常 | Docker 怎么部署 | 0.58 |
| 无关 | 今天天气 | 1.27 |
| 无关 | 红烧肉 | 1.30 |
| 无关 | 股市 | 1.34 |
| 乱码 | asdfghjkl | 1.15 |

**选 1.1 的原因**：
- 正常问题最高 0.89，还有 0.21 余量
- 乱码 1.15 也能被拦住
- 无关问题 1.27+ 都被拦住

**为什么用向量距离，不用 RRF 分数**：
- RRF 分数范围窄（0.005~0.02），无法区分好坏
- 向量 L2 距离直接反映语义相似度，范围 0~2，好判断

## 六、遇到的坑（面试重点）

| 坑 | 原因 | 解决方案 |
|---|---|---|
| DeepSeek 不支持 json_schema | 模型 API 限制 | 改用 PydanticOutputParser，本地解析 |
| 中文引号混入字符串 | 复制时带入全角引号 | 统一用英文半角引号 |
| fastmcp 4.x 与 langchain-mcp-adapters 冲突 | mcp 版本要求矛盾（<2.0 vs >=2.0） | 降级 fastmcp 到 3.4.7 |
| HuggingFace 联网卡住 | 国内无法访问 huggingface.co | 设 HF_HUB_OFFLINE=1，用本地缓存路径 |
| uvicorn --reload 下调 asyncio.run 报错 | 事件循环嵌套 | 改用 FastAPI lifespan 异步钩子 |
| MCP 工具同步调用报错 | MCP 工具是异步的 | Agent 改用 astream / ainvoke |
| stream 输出吞进中间步骤 | 工具调用前的英文思考也被流式输出 | 记录 tool 节点位置，只输出它之后的内容 |
| Docker 构建慢 | 模型下载 + 依赖安装 | 分层缓存，模型下载放独立层 |
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

### 3. 怎么防止 RAG 幻觉？

**两层防护**：

**第一层（Prompt 软约束）**：
- Prompt 里加"只使用资料中的信息，不要编造"

**第二层（检索层硬降级）**：
- 用向量距离做阈值，超过就返回"知识库中没有相关信息"
- 不调 LLM，从根源上避免幻觉
- 加评估集，量化回答准确率

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

### 7. 并发量高怎么优化？

- Agent 改用异步（astream / ainvoke）
- MCP Server 常驻，不用每次启动
- 向量库换 FAISS / Chroma 持久化版本
- 加 Redis 限流（incr + expire）
- 多 worker 部署（uvicorn --workers N 或 gunicorn）
- 反向代理（Nginx）分担负载

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

**为什么需要**：
- 纯向量检索对关键词不敏感
- 用户用口语化表达时，和文档术语有语义鸿沟
- BM25 补上关键词匹配能力

### 11. 怎么防止 Agent 死循环？

**两道护栏**：

1. **`recursion_limit=10`**：Agent 最多循环 10 次，超过抛异常
2. **`max_tokens=2000`**：单次输出最多 2000 token

**在代码里怎么用**：

```python
config = {"recursion_limit": RECURSION_LIMIT}
result = await agent.ainvoke({...}, config=config)
```

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
    checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        agent = create_agent(
            model, tools=tools,
            checkpointer=saver,
            system_prompt="..."
        )
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

**只需要传当前这一轮的消息**，Agent 会自动从 checkpointer 加载历史。

**测试验证**：
1. `test_001` 问"MMR 是什么？" → 正常回答
2. **重启服务**
3. `test_001` 问"它和普通相似度检索有什么区别？" → 仍能理解"它"指 MMR ✅

**生产优化**：AsyncSqliteSaver 换 `PostgresSaver`，支持多实例共享。

**面试话术**：
> "我的记忆用 LangGraph 的 checkpointer 做唯一真相源，持久化用 AsyncSqliteSaver（生产环境会换 PostgresSaver）。每次请求只传当前这一轮的消息，Agent 自动从 checkpointer 加载该 thread_id 的历史。之前我试过手动从 MySQL 加载历史拼进 messages，但发现和 checkpointer 的 state 会打架——消息格式混用、历史可能重复累积。后来统一到 checkpointer，MySQL 只做展示层给 /history 接口用。"

### 13. 检索不到内容怎么处理？

**降级策略**：

1. **检索层**：用向量距离做阈值（`MAX_DISTANCE = 1.1`），超过就返回空
2. **工具层**：收到空结果时返回"知识库中没有找到相关信息"
3. **不调 LLM**：从根源上避免模型基于无关片段产生幻觉

**阈值怎么定**：
- 实测正常问题距离 0.58~0.89
- 实测无关问题距离 1.15~1.34
- 选 1.1 能分开两类

**面试话术**：
> "我在检索层加了降级策略。用相似度距离做阈值，超过阈值就返回'知识库中没有相关信息'，不调 LLM。这样避免模型在缺乏依据时产生幻觉。阈值是通过实测校准的，我测了正常问题和无关问题的距离分布，选了能分开两类的最优值。"

### 14. RAG 评估用什么指标？

**四个指标**：

| 指标 | 公式 | 含义 |
|---|---|---|
| 召回率 | 命中数 / 总数 | 该找的有没有找到 |
| 精确率 | 平均（相关结果数 / Top-K 总数） | 找到的有多少是对的 |
| MRR | 平均（1 / 第一个正确结果的排名） | 第一个正确结果排多前 |
| 多样性 | 平均（Top-K 里不同文档数） | 结果覆盖多少不同来源 |

**实测数据**（30 个测试项）：

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 相似度 | 100% | 0.87 | 0.96 | 1.37 |
| MMR | 100% | 0.83 | 0.97 | 1.43 |
| MMR + 去重 | 100% | 0.77 | 0.97 | 1.57 |

**面试话术**：
> "我评估 RAG 用了四个指标。召回率看'该找的有没有找到'，精确率看'找到的有多少是对的'，MRR 看'第一个正确结果排多前'，多样性看'结果覆盖多少不同文档'。实测发现精确率和多样性呈 trade-off：追求精确用相似度，需要多角度用 MMR+去重。MRR 三种策略都在 0.96 以上，Top-1 都很准。"

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

**服务端**：完整异常打日志，方便排查
**客户端**：只看到友好提示，不泄漏内部信息

**面试话术**：
> "服务端用完整的 `str(e)` 打日志，客户端只返回 `friendly_error` 映射后的友好提示。因为 `str(e)` 可能包含 API key、文件路径、SQL 结构等内部信息，生产环境不能直接暴露给客户端。我用异常类型 + HTTP 状态码判断，而不是子串匹配——因为 `rate` 会误匹配 `generate`。"

### 16. 如果让你重做，会改什么？

- 用 uv + 虚拟环境管理依赖，避免全局冲突
- RAG 加 Rerank（bge-reranker-base）
- 加 API Key 鉴权和限流
- 加 LangSmith 可观测性
- 写单元测试和集成测试
- 文档库扩充到几百个文件，让不同检索策略的差异更明显
- 加 chunk 级评估（不只是文档级）

## 八、项目数据（面试时能报的具体数字）

- 工具数量：1 个（search_knowledge）
- 知识库文档数：2 个 Markdown 文件（真实笔记）
- 切分后块数：102 块
- Embedding 模型：BAAI/bge-small-zh-v1.5（512 维）
- 向量库：FAISS（本地持久化）
- 检索策略：混合检索（向量 + BM25 + 加权 RRF）
- RRF 参数：k=60, vector_weight=0.7, bm25_weight=0.3
- 降级阈值：MAX_DISTANCE=1.1
- 护栏参数：recursion_limit=10, max_tokens=2000
- 记忆：LangGraph checkpointer + AsyncSqliteSaver（SQLite 文件，重启不丢）
- 展示层：MySQL（表 conversations）
- 评估指标：召回率 / 精确率 / MRR / 多样性（30 个测试项）
- 首次 Docker 构建耗时：约 19 分钟
- 单次请求耗时：2~3 秒
- 镜像大小：10.4 GB（实际磁盘占用 3.46 GB）

## 九、可演示的操作

1. 打开 Swagger UI：http://127.0.0.1:8000/docs
2. 测普通对话：POST /chat
3. 测流式对话：POST /chat/stream
4. 查看对话历史：GET /history?thread_id=user_001
5. **演示持久化记忆**：问"MMR 是什么" → **重启服务** → 问"它和普通相似度有什么区别" → 仍能理解
6. 演示会话隔离：换 thread_id 问"它是什么"
7. 演示降级策略：问"今天天气怎么样？" → 返回"知识库中没有找到相关信息"
8. 用浏览器打开 client.html，看打字机效果
9. 展示 Docker 一键启动：docker compose up -d
10. 展示 GitHub 仓库：https://github.com/wzy106/rag-agent-service

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
| bge-small-zh-v1.5 | 512 | 轻量，中文好，我用这个 |
| bge-base-zh-v1.5 | 768 | 效果更好，慢一点 |
| bge-large-zh-v1.5 | 1024 | 效果最好，吃内存 |
| text-embedding-3-small | 1536 | OpenAI 官方，需付费 |

### Rerank 是什么

- 向量检索：双塔模型，快但粗，用于粗排
- Rerank：交叉编码器，慢但精，用于精排
- 流程：向量检索 Top-10 → Rerank → 取 Top-3
- 常用模型：BAAI/bge-reranker-base

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

## 十一、MySQL 展示层 + Redis 预留（工程化）

### 为什么还需要 MySQL

| 问题 | 解法 |
|---|---|
| 用户要查看历史对话 | MySQL 展示层，给 /history 接口用 |
| Agent 的记忆 | 交给 LangGraph checkpointer，不经过 MySQL |

**角色划分**：
- **checkpointer（SQLite）**：Agent 的记忆，唯一真相源
- **MySQL**：只做展示层，给 `/history` 接口用
- **Redis**：暂时不用，预留做限流

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

### Redis 预留说明

**当前没用**。历史版本用过 Redis 缓存，但实测发现：
- 缓存 key 用 `context_version` 做版本号
- `context_version` = MySQL 消息数，每轮 +2
- 同一问题第二次必然 miss，**命中率结构性为 0**
- 是负优化，已删除

**下一步计划**：
- 用于限流（`incr` + `expire`）
- 或用于分布式缓存（多实例共享）

**为什么保留依赖**：
- `requirements.txt` 和 `docker-compose.db.yml` 里还留着
- 将来加限流时不用重新配置

### 完整数据流（当前版本）

```text
用户提问（带 thread_id）
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
| 为什么不删 Redis 依赖？ | 预留做限流（incr + expire） |
| 多实例部署怎么共享状态？ | checkpointer 换 PostgresSaver；限流用 Redis |

### 可优化方向

- Redis 加密码和连接池，生产环境必须
- 用连接池代替每次新建 MySQL 连接（性能优化）
- MySQL 写入改异步（aiomysql）

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
    checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
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

**方案**：用 `AsyncSqliteSaver` 做持久化 checkpointer。

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
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

### 10. 面试高频问题

| 问题 | 答案要点 |
|---|---|
| Pydantic 在 FastAPI 里做什么？ | 请求/响应模型、类型校验、自动文档 |
| lifespan 是什么？ | 启动和关闭钩子，替代废弃的 on_event |
| 为什么用 lifespan 不用顶层代码？ | 顶层会和 uvicorn 事件循环冲突 |
| 流式接口怎么过滤中间步骤？ | `saw_tool` 标志分界，`pending` 缓存工具前内容 |
| SSE 格式是什么？ | `data: 内容\n\n`，结束发 `[DONE]` |
| 为什么 [DONE] 不能放 finally？ | 客户端断连时 aclose() 抛 GeneratorExit，finally 里 yield 会抛 RuntimeError |
| Agent 死循环怎么防？ | `recursion_limit=10` + `max_tokens=2000` |
| Agent 多轮对话怎么实现？ | LangGraph checkpointer + AsyncSqliteSaver |
| 错误信息怎么处理？ | `friendly_error` 映射，用异常类型判断 |
| 检索不到内容怎么处理？ | 距离阈值降级，返回"没找到"，不调 LLM |
| RAG 评估用什么指标？ | 召回率 + 精确率 + MRR + 多样性 |

## 十三、RAG 评估（面试重点）

### 评估方法

构建 30 个测试项的测试集，每个标注期望命中的文档。
问题故意不包含文档关键词，测试语义检索能力。
包含单文档问题和多文档问题（一个 query 对应多个相关文档）。

### 评估脚本核心

```python
def evaluate(search_fn, name, k=3):
    hit = 0
    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]
        if any(exp in sources for exp in item["expected_sources"]):
            hit += 1
        else:
            print(f"❌ {item['question']} → 期望 {item['expected_sources']}，实际 {sources}")
    recall = hit / len(test_set)
    print(f"[{name}] 召回率：{hit}/{len(test_set)} = {recall:.0%}")

def evaluate_full(search_fn, name, k=3):
    hit = 0
    precision_sum = 0
    mrr_sum = 0

    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]

        if any(exp in sources for exp in item["expected_sources"]):
            hit += 1

        if sources:
            relevant = sum(1 for s in sources if s in item["expected_sources"])
            precision_sum += relevant / len(sources)

        for rank, s in enumerate(sources, 1):
            if s in item["expected_sources"]:
                mrr_sum += 1 / rank
                break

    n = len(test_set)
    print(f"[{name}]")
    print(f"  召回率：{hit}/{n} = {hit/n:.0%}")
    print(f"  精确率：{precision_sum/n:.2f}")
    print(f"  MRR：{mrr_sum/n:.2f}")
```

### 评估结果（当前 2 个文档 / 102 chunk）

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 100% | **0.87** | 0.96 | 1.37 |
| MMR | 100% | 0.83 | **0.97** | 1.43 |
| MMR + 按来源去重 | 100% | 0.77 | **0.97** | **1.57** |

### 关键发现

**1. 精确率和多样性呈明显 trade-off**

- 相似度：精确率最高（0.87），多样性最低（1.37）
- MMR + 去重：多样性最高（1.57），精确率最低（0.77）

**2. MRR 几乎不变（0.96~0.97）**

三种策略的 Top-1 都很准。

**3. 选哪个策略取决于业务**

| 场景 | 推荐策略 |
|---|---|
| 追求精确（如事实查询） | 相似度 |
| 需要多角度（如分析任务） | MMR + 去重 |
| 平衡 | MMR |

### 早期评估（5 个文档 / 24 chunk）

| 策略 | 召回率 | 平均多样性 |
|---|---|---|
| 普通相似度 | 29/30 = 97% | 2.00 |
| MMR | 28/30 = 93% | 2.50 |
| MMR + 按来源去重 | — | 3.00 |

**核心结论**：MMR 用 4% 的召回率换取 25% 的多样性提升。

### 失败案例分析

**案例 1**："怎么让回答一个字一个字显示出来？"
- 期望：`deployment.md`
- 实际：`langchain.md`、`rag.md`
- 原因：口语化表达和文档术语有语义鸿沟

**案例 2**："多 Agent 系统怎么保证消息不丢？"
- 期望：`langgraph.md`、`deployment.md`
- 实际：`langchain.md`、`rag.md`
- 原因：MMR 强制 chunk 级多样性时挤掉了最相关的文档

### 已知不足

**评估层面**：
- **文档级召回而非 chunk 级**：用 `source`（文件名）判断命中，只有两三篇文档，命中太容易，含金量不高。生产环境应该标注 chunk id 或关键句。
- **30 条样本太少**：3% 的差异（97% vs 93%）是一条样本的差别，统计上无意义。应该扩到 200 条以上。
- **没有端到端指标**：检索对了不等于回答对了。应该加 LLM-as-judge 或人工标注的 answer correctness。
- **随机基线**：当前随机检索的召回率是 84.1%，三种策略全 100%，**判别力不足**。

**架构层面**：
| 位置 | 问题 | 生产级解法 |
|---|---|---|
| `/chat` 和 `/chat/stream` | pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环 | 用 `aiomysql` |
| `save_conversation` | 每次请求新建连接，没有连接池 | 用连接池（如 `DBUtils`） |
| `MAX_DISTANCE = 1.1` | 阈值校准不够系统，是拍脑袋定的 | 对正负样本统计距离分布，取分离点 |
| 流式状态机 | 用三个布尔变量表达状态，边界情况可能出错 | 改用 `metadata["langgraph_node"]` 做来源判断 + 显式状态机 |
| MCP Server 用了私有 API | `vectorstore.docstore._dict` 是内部 API，版本升级可能坏 | 构建索引时自己维护 chunks 列表并 pickle 落盘 |
| Docker 部署 | Dockerfile 缺 COPY docs/，compose 缺 network，host 写死 localhost | 待修 |

### 改进方案

| 方案 | 原理 | 状态 |
|---|---|---|
| 混合检索 | 向量 + BM25，加权合并 | ✅ 已做 |
| 降级策略 | 距离阈值拦截无关问题 | ✅ 已做 |
| 多指标评估 | 召回率 + 精确率 + MRR + 多样性 | ✅ 已做 |
| 友好错误提示 | friendly_error 映射（异常类型） | ✅ 已做 |
| 删掉无效缓存 | 命中率结构性为 0 | ✅ 已做 |
| Rerank 精排 | bge-reranker-base 对 Top-20 重排 | ⏳ 待做 |
| Chunk 级评估 | 标注 chunk id，不只标文档 | ⏳ 待做 |
| 扩充测试集 | 200 条以上 + 置信区间 | ⏳ 待做 |
| 端到端评估 | LLM-as-judge 或人工标注 | ⏳ 待做 |
| 修 Docker 部署 | COPY docs/ + network + 服务名 host | ⏳ 待做 |

### 面试答题模板

> "我构建了 30 个测试项的 RAG 评估集，故意让问题不含文档关键词。用了四个指标：召回率、精确率、MRR、多样性。实测发现精确率和多样性呈 trade-off：纯相似度精确率 0.87、多样性 1.37；MMR+去重多样性 1.57、精确率 0.77。MRR 三种策略都在 0.96 以上，Top-1 都很准。我还算过随机基线——从 102 个 chunk 里随便抽 3 个，召回率 84.1%。这说明我的测试集判别力不够，下一步要扩到 200 条 + chunk 级标注。"

## 十四、项目新增数据

- MySQL 表名：`conversations`（展示层）
- MySQL 字段：id, thread_id, role, content, created_at
- 新增接口：`GET /history?thread_id=user_001&limit=20`
- 评估集规模：30 个测试项
- 当前知识库：2 个文档、102 个 chunk
- 评估指标：召回率 / 精确率 / MRR / 多样性
- 随机检索基线：84.1%
- 相似度：召回率 100%，精确率 0.87，MRR 0.96，多样性 1.37
- MMR：召回率 100%，精确率 0.83，MRR 0.97，多样性 1.43
- MMR + 去重：召回率 100%，精确率 0.77，MRR 0.97，多样性 1.57
- 混合检索参数：rrf_k=60, vector_weight=0.7, bm25_weight=0.3
- 降级阈值：MAX_DISTANCE=1.1
- 护栏参数：recursion_limit=10, max_tokens=2000
- 记忆：LangGraph checkpointer + AsyncSqliteSaver（SQLite 文件，重启不丢）
- 缓存：暂时没有（已删 Redis，命中率结构性为 0）
- Redis：依赖保留，预留做限流
- 错误处理：`friendly_error` 用异常类型映射，不泄漏内部信息
- requirements.txt：18 个包，已在干净 venv 里验证可导入
- GitHub 仓库：https://github.com/wzy106/rag-agent-service