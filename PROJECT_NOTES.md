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
    │  混合检索：                       │
    │  向量检索 Top-9  +  BM25 Top-9   │
    │       ↓                          │
    │  加权 RRF 合并（向量0.7+BM25 0.3）│
    │       ↓                          │
    │  Top-3 返回                       │
    └─────────────────────────────────┘
                   ↓
    检索结果返回给 Agent（循环第 ④ 步）
                   ↓ SSE 流式返回
客户端
```

## 三、技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | LangChain 1.4 + LangGraph 1.2 |
| 模型 | DeepSeek（OpenAI 兼容协议） |
| 工具协议 | MCP（FastMCP 3.4.7） |
| 检索 | sentence-transformers + FAISS + BM25 + 加权 RRF |
| 缓存 | Redis |
| 历史 | MySQL |
| 服务 | FastAPI + Uvicorn |
| 部署 | Docker + Docker Compose |
| 流式 | SSE（Server-Sent Events） |

## 四、核心功能

- 多轮对话（InMemorySaver + thread_id）
- RAG 检索增强（本地 embedding + FAISS 向量库）
- 混合检索（向量 + BM25 + 加权 RRF）
- MCP 工具解耦（Agent 和工具分进程）
- SSE 流式输出（打字机效果）
- 中间步骤过滤（只输出最终回答）
- **Agent 护栏（recursion_limit=10，max_tokens=2000）**
- 异常处理（错误以 SSE 格式返回）
- 请求日志（收到消息 + 耗时统计）
- FAISS 索引持久化（避免重复 embedding）
- Redis 缓存高频问题（省 Token）
- MySQL 持久化对话历史
- RAG 评估（召回率 + 多样性对比）

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

- Prompt 里加"只使用资料中的信息，不要编造"
- 检索结果不够相关时，返回"知识库中没有相关信息"
- 加 Rerank 二次排序
- 加评估集，量化回答准确率

### 4. InMemorySaver 和 PostgresSaver 区别？

- InMemorySaver：存内存，进程结束即丢失，适合开发测试
- PostgresSaver：存数据库，持久化，多实例共享，适合生产

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
- 加 Redis 缓存高频问题
- 加限流（SlowAPI）
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

### 12. 如果让你重做，会改什么？

- 用 uv + 虚拟环境管理依赖，避免全局冲突
- RAG 加 Rerank（bge-reranker-base）
- 加 API Key 鉴权和限流
- 加 LangSmith 可观测性
- 写单元测试和集成测试
- 文档库扩充到几百个文件，让不同检索策略的差异更明显

## 八、项目数据（面试时能报的具体数字）

- 工具数量：1 个（search_knowledge）
- 知识库文档数：2 个 Markdown 文件（真实笔记）
- 切分后块数：102 块
- Embedding 模型：BAAI/bge-small-zh-v1.5（512 维）
- 向量库：FAISS（本地持久化）
- 检索策略：混合检索（向量 + BM25 + 加权 RRF）
- RRF 参数：k=60, vector_weight=0.7, bm25_weight=0.3
- **护栏参数：recursion_limit=10, max_tokens=2000**
- 缓存：Redis（TTL 3600 秒）
- 历史：MySQL（表 conversations）
- 首次 Docker 构建耗时：约 19 分钟
- 单次请求耗时：未命中 2~3 秒，命中 <0.1 秒
- 镜像大小：10.4 GB（实际磁盘占用 3.46 GB）

## 九、可演示的操作

1. 打开 Swagger UI：http://127.0.0.1:8000/docs
2. 测普通对话：POST /chat
3. 测流式对话：POST /chat/stream
4. 查看对话历史：GET /history?thread_id=user_001
5. 用浏览器打开 client.html，看打字机效果
6. 展示 Docker 一键启动：docker compose up -d
7. 展示 GitHub 仓库：https://github.com/wzy106/rag-agent-service

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

| 指标 | 含义 |
|---|---|
| 召回率 | 相关文档有没有被检索到 |
| 精确率 | 检索到的有多少是相关的 |
| 忠实度 | 回答是否忠于检索结果 |
| 答案相关性 | 回答是否切题 |

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

## 十一、Redis 缓存 + MySQL 历史（工程化）

### 为什么需要这两个

| 问题 | 解法 |
|---|---|
| 同一个问题反复问，每次调大模型浪费 Token | Redis 缓存回答 |
| 服务重启后对话历史丢失 | MySQL 持久化 |
| 多实例部署状态不一致 | MySQL + Redis 共享外部存储 |

### Redis 缓存实现

```python
import redis, hashlib

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_cached(query: str):
    key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    return r.get(key)

def set_cache(query: str, answer: str, ttl: int = 3600):
    key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    r.set(key, answer, ex=ttl)
```

**关键点**：
- key 用 query 的 MD5，避免中文和特殊字符
- TTL 设 1 小时，过期自动清理
- 命中缓存不调大模型，省 Token 也省时间

### MySQL 对话历史实现

```sql
CREATE TABLE conversations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    thread_id VARCHAR(64),
    role VARCHAR(20),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

```python
import pymysql

def save_conversation(thread_id: str, role: str, content: str):
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password=os.getenv("MYSQL_PASSWORD", "root123"), database="agent", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, role, content)
            )
        conn.commit()
    finally:
        conn.close()
```

**关键点**：
- 按 thread_id 分组，不同用户隔离
- 用 `%s` 参数化查询，防 SQL 注入
- `charset="utf8mb4"` 支持中文和 emoji
- `try/finally` 保证连接一定关闭

### 完整数据流

```text
用户提问
    ↓
Redis 查缓存
    ├── 命中 → 返回缓存 + 写 MySQL 历史
    └── 未命中 → 调 Agent → 写 Redis 缓存 + 写 MySQL 历史 → 返回
```

### 面试答题要点

| 问题 | 答案 |
|---|---|
| 高频问题怎么优化？ | Redis 缓存，key 是 query 的 MD5，TTL 一小时 |
| 对话历史怎么持久化？ | MySQL 存，按 thread_id 分组 |
| 多实例部署怎么共享状态？ | MySQL + Redis 都放外部，实例无状态 |
| Redis 和 MySQL 区别？ | Redis 内存、快、适合缓存；MySQL 磁盘、持久、适合结构化数据 |
| 限流怎么做？ | Redis `incr` + `expire`，每分钟超 N 次返回 429 |
| 为什么缓存 key 用 MD5？ | 避免中文/特殊字符问题，固定长度省内存 |

### 可优化方向

- 缓存加版本号（`chat:v1:xxx`），prompt 改了自动失效
- 历史只加载最近 N 轮，避免上下文太长
- Redis 加密码和连接池，生产环境必须
- 用连接池代替每次新建连接（性能优化）

## 十二、FastAPI 服务层设计（面试重点）

### 1. Pydantic 请求/响应模型

```python
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
```

**作用**：
- 定义请求体和响应体的结构
- FastAPI 自动做类型校验，字段错/类型错返回 422
- 自动生成 Swagger UI 文档
- IDE 有自动补全

**对比 `dict` 写法**：
- `dict`：无校验、文档模糊、无补全
- Pydantic：强类型、文档清晰、开发体验好

### 2. lifespan 生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑
    agent = create_agent(...)
    ensure_table()
    yield
    # 关闭逻辑
    print("[关闭] 服务停止")

app = FastAPI(lifespan=lifespan)
```

**为什么不用模块顶层？**
- 顶层代码在 import 时就执行，FastAPI 事件循环还没启动
- 在顶层调 `asyncio.run()` 会和 uvicorn 事件循环冲突
- lifespan 在事件循环里执行，可以 `await`

**适合放什么**：
- 连接数据库 / Redis / MCP Server
- 加载模型
- 建表（幂等）

### 3. MySQL 幂等建表

```python
def ensure_table():
    conn = pymysql.connect(...)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    thread_id VARCHAR(64),
                    role VARCHAR(20),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
    finally:
        conn.close()
```

**关键点**：
- `IF NOT EXISTS`：表存在则跳过，**幂等**
- 在 lifespan 启动时调用，**服务自愈**（数据卷被清空也能自动重建）
- `try/finally` 保证连接一定关闭

### 4. SSE 流式接口的中间步骤过滤

**问题**：Agent 在调用工具前会有英文思考（"I'll look that up..."），不该给用户看。

**解法**：用 `saw_tool` 标志区分中间步骤和最终回答。

```python
saw_tool = False    # 是否已过工具节点
pending = []        # 工具调用前的内容（先缓存）
started = False     # 是否已开始输出最终回答

async for chunk in agent.astream(...):
    msg_chunk, metadata = chunk

    if msg_chunk.type == "tool":
        saw_tool = True
        pending = []        # 清空中间步骤
        continue

    if not msg_chunk.content:
        continue

    if saw_tool:
        # 工具已返回，这是最终回答，实时输出
        yield f"data: {json.dumps({'text': msg_chunk.content})}\n\n"
        started = True
    else:
        # 工具调用前的内容，先缓存
        pending.append(msg_chunk.content)

# 整个流程没调工具（纯聊天），把缓存一次性输出
if not started and pending:
    for text in pending:
        yield f"data: {json.dumps({'text': text})}\n\n"
```

**三个变量的作用**：

| 变量 | 作用 |
|---|---|
| `saw_tool` | 分界线，工具返回前后 |
| `pending` | 缓存工具调用前的内容 |
| `started` | 标记是否已输出最终回答 |

**设计权衡**：
- RAG 场景（有工具调用）：完全正常，实时流式
- 纯聊天场景（无工具）：先缓存再一次性输出，打字机效果打折

### 5. SSE 格式规范

```python
yield f"data: {json.dumps({'text': msg_chunk.content}, ensure_ascii=False)}\n\n"
```

| 部分 | 作用 |
|---|---|
| `data: ` | SSE 协议要求的前缀 |
| `json.dumps(...)` | 转成 JSON，方便客户端解析 |
| `ensure_ascii=False` | 保留中文，不转 `\uXXXX` |
| `\n\n` | 两条换行表示"本条消息结束" |

**结束标记**：
```python
yield "data: [DONE]\n\n"
```
告诉客户端"流结束了"，客户端停止读取。

### 6. 异常处理

```python
try:
    ...
except Exception as e:
    yield f"data: {json.dumps({'error': f'服务出错了：{str(e)}'})}\n\n"
finally:
    yield "data: [DONE]\n\n"
```

**好处**：
- 不返回 HTTP 500 一大坨堆栈
- 客户端拿到结构化的错误信息
- `finally` 保证 `[DONE]` 一定发出

### 7. Redis 缓存 key 设计

```python
key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
```

**为什么用 MD5？**
- 中文/标点直接当 key 会出问题
- 长 query 当 key 浪费内存
- MD5 固定 32 字符，安全稳定

**为什么加 `chat:` 前缀？**
- 命名空间隔离，Redis 里可能有其他 key
- 方便用 `KEYS chat:*` 批量查看

### 8. Agent 护栏（防止死循环和 Token 爆炸）

**问题**：Agent 在复杂任务下可能无限调用工具，或者单次输出过长，浪费 Token。

**方案**：两道护栏。

```python
RECURSION_LIMIT = 10    # 最大循环次数
MAX_TOKENS = 2000       # 单次输出上限

model = ChatOpenAI(
    ...,
    max_tokens=MAX_TOKENS,   # 护栏 1：限制输出长度
)

agent = create_agent(model, tools=tools, ...)

# 护栏 2：每次调用带 recursion_limit
config = {"recursion_limit": RECURSION_LIMIT}
result = await agent.ainvoke({...}, config=config)
async for chunk in agent.astream({...}, config=config, stream_mode="messages"):
    ...
```

**两道护栏的作用**：

| 护栏 | 防止什么 | 触发后 |
|---|---|---|
| `recursion_limit=10` | Agent 死循环（无限调工具） | 抛 `GraphRecursionError`，被 `try/except` 捕获，返回友好错误 |
| `max_tokens=2000` | 单次输出过长，浪费 Token | 模型自然截断，最多输出 2000 token |

**面试话术**：
> "我给 Agent 设了两道护栏：一是 recursion_limit=10，超过自动抛异常终止；二是 max_tokens=2000，限制单次输出。第一道防止 Agent 无限调用工具，第二道防止单次回答过长。在流式接口里，异常会被 try/except 捕获，以 SSE 格式返回友好错误，不会让服务崩掉。"

### 9. 面试高频问题

| 问题 | 答案要点 |
|---|---|
| Pydantic 在 FastAPI 里做什么？ | 请求/响应模型、类型校验、自动文档 |
| lifespan 是什么？ | 启动和关闭钩子，替代废弃的 on_event |
| 为什么用 lifespan 不用顶层代码？ | 顶层会和 uvicorn 事件循环冲突 |
| 流式接口怎么过滤中间步骤？ | `saw_tool` 标志分界，`pending` 缓存工具前内容 |
| SSE 格式是什么？ | `data: 内容\n\n`，结束发 `[DONE]` |
| MySQL 表怎么初始化？ | lifespan 里 `CREATE TABLE IF NOT EXISTS`，幂等自愈 |
| Agent 死循环怎么防？ | `recursion_limit=10` + `max_tokens=2000` |

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

def measure_diversity(search_fn, name, k=3):
    total_docs = 0
    for item in test_set:
        results = search_fn(item["question"], k)
        sources = set(doc.metadata["source"] for doc in results)
        total_docs += len(sources)
    print(f"[{name}] 平均多样性：{total_docs / len(test_set):.2f} 个不同文档/查询")
```

### 评估结果（当前 2 个文档 / 102 chunk）

| 策略 | 召回率 |
|---|---|
| 普通相似度 | 30/30 = 100% |
| MMR | 30/30 = 100% |
| 混合检索（等权 RRF 0.5/0.5） | 29/30 = 97% |
| 混合检索（加权 RRF 0.7/0.3） | 30/30 = 100% |

### 关键发现

**1. 等权 RRF 反而更差（97%）**

失败案例："PydanticOutputParser 怎么用？"
- 期望：`my_langchain_notes.md`
- 实际：`my_project_notes.md` × 3
- 原因：BM25 看到关键词就把它排前面，两路等权时错误被放大

**2. 加权 RRF 解决（100%）**

- 向量权重 0.7，BM25 权重 0.3
- 让更可靠的一路（向量检索）主导
- 3% 的错误被消除

**3. 文档规模小，差异不明显**

- 只有 2 个文档、102 个 chunk
- 三种策略都是 100%（除了等权混合）
- 如果文档量级达到几百上千，差异会更明显

### 早期评估（5 个文档 / 24 chunk）

**这组数据用于对比 MMR 和相似度，体现"多样性 vs 召回率"的权衡：**

| 策略 | 召回率 | 平均多样性 |
|---|---|---|
| 普通相似度 | 29/30 = 97% | 2.00 |
| MMR | 28/30 = 93% | 2.50 |
| MMR + 按来源去重 | — | 3.00 |

**核心结论**：MMR 用 4% 的召回率换取 25% 的多样性提升。

### 失败案例分析

**案例 1**："怎么让回答一个字一个字显示出来？"
- 期望：`deployment.md`（流式输出那节）
- 实际：`langchain.md`、`rag.md`
- 原因：用户用口语表达，文档用术语"流式输出"、"SSE"

**案例 2**："多 Agent 系统怎么保证消息不丢？"
- 期望：`langgraph.md`、`deployment.md`
- 实际：`langchain.md`、`rag.md`、`langchain.md`
- 原因：MMR 强制 chunk 级多样性时挤掉了最相关的文档

### 改进方案

| 方案 | 原理 | 成本 |
|---|---|---|
| 混合检索 | 向量 + BM25，加权合并 | 中（已做） |
| 查询改写 | LLM 把口语改写成术语 | 低 |
| 文档增强 | 补充口语化同义表达 | 低 |
| Rerank | 先粗排 Top-20，再精排 Top-3 | 中 |

### 面试答题模板

> "我构建了 30 个测试项的 RAG 评估集，故意让问题不含文档关键词。对比了四种策略：相似度 100%、MMR 100%、等权混合检索 97%、加权混合检索 100%。等权 RRF 反而变差，因为 RRF 奖励两路都出现的文档，BM25 判断错会被放大。加权后（向量 0.7、BM25 0.3）问题解决。这告诉我：混合检索不一定比纯向量好，关键在于权重调优。"

### 可优化方向

- 加 Rerank 精排
- 扩充测试集到 100 题以上
- 引入 RAGAS 自动评估生成质量（忠实度、答案相关性）
- 文档库扩充到几百个文件，让策略差异更明显

## 十四、项目新增数据

- Redis key 格式：`chat:{md5}`
- Redis TTL：3600 秒
- MySQL 表名：`conversations`
- MySQL 字段：id, thread_id, role, content, created_at
- 缓存命中 vs 未命中：未命中约 2~3 秒，命中不到 0.1 秒
- 新增接口：`GET /history?thread_id=user_001&limit=20`
- 评估集规模：30 个测试项
- 当前知识库：2 个文档、102 个 chunk
- 相似度召回率：100%
- MMR 召回率：100%
- 等权混合检索：97%
- 加权混合检索：100%
- 混合检索参数：rrf_k=60, vector_weight=0.7, bm25_weight=0.3
- **护栏参数：recursion_limit=10, max_tokens=2000**
- GitHub 仓库：https://github.com/wzy106/rag-agent-service