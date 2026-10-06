# 项目踩坑与复盘

> 记录项目开发过程中遇到的问题、分析过程和解决方案。
> 每个故事按"问题 → 分析 → 方案 → 收获"结构。

## 快速索引

| 类别 | 对应故事 |
|---|---|
| 模型 API 限制 | 故事 1 |
| 依赖冲突 | 故事 2 |
| 网络问题 | 故事 3 |
| 异步编程 | 故事 4、5 |
| 用户体验 | 故事 6 |
| 构建优化 | 故事 7 |
| 协议理解 | 故事 8 |
| 检索算法 | 故事 9、10 |
| 生产级意识 | 故事 11、13 |
| 功能演进回归 | 故事 12 |
| 幻觉治理 | 故事 14 |
| 评估方法 | 故事 15 |
| 记忆持久化 | 故事 16 |

---

## 故事 1：DeepSeek 不支持 json_schema

### 问题
做结构化输出时，用 LangChain 的 `with_structured_output` 报错：
```text
Error code: 400 - This response_format type is unavailable now
```

### 分析
LangChain 默认用 `json_schema` 模式强制模型输出结构化数据，但 DeepSeek 的 API 不支持这个模式。

### 方案
改用 `PydanticOutputParser`：
1. `parser.get_format_instructions()` 生成格式说明
2. 通过 `.partial()` 注入 system prompt
3. 模型输出 JSON 字符串
4. `parser` 本地解析为 Pydantic 对象

### 为什么这么选
- **兼容性最好**：不依赖模型 API 对 `json_schema` 的支持
- **原理透明**：提示词工程 + 本地解析，完全可控
- **可切换**：如果将来换模型支持 `json_schema`，可以随时切回

### 收获
模型 API 的差异需要用适配层来屏蔽，不能假设所有模型都支持相同特性。

---

## 故事 2：fastmcp 和 langchain-mcp-adapters 版本冲突

### 问题
装 `langchain-mcp-adapters` 后，原本能跑的 `fastmcp` 突然报错：
```text
ImportError: cannot import name 'AuthorizationCodeResult' from 'mcp.shared.auth'
```

### 分析
- `langchain-mcp-adapters` 要求 `mcp<2.0.0`
- `fastmcp 4.0.10` 要求 `mcp>=2.0.0`
- pip 把 `mcp` 从 2.2.0 降级到 1.30.0，导致 fastmcp 坏了

### 方案
降级 fastmcp：
```cmd
pip install "fastmcp<4.0"
```
装到 `fastmcp 3.4.7`，和 `mcp 1.30.0` 兼容。

### 为什么这么选
- **降级成本低**：3.4.7 的 API 和 4.x 几乎一样，代码不用改
- **生态兼容**：`langchain-mcp-adapters` 是官方的，不能换
- **长期方案**：用虚拟环境隔离不同项目的依赖

### 收获
全局环境下的依赖冲突是常见问题。生产项目必须用虚拟环境隔离依赖。

---

## 故事 3：HuggingFace 联网卡住

### 问题
跑 RAG 时，加载本地 embedding 模型卡住：
```text
[WinError 10060] 由于连接方在一段时间后没有正确答复...
Retrying in 1s [Retry 1/5].
```

### 分析
`SentenceTransformer` 默认会去 huggingface.co 检查模型更新。国内网络连不上，重试 5 次后崩溃。

### 方案
设两个环境变量：
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```
强制离线，只用本地缓存。

### 为什么这么选
- **本地已有模型**：之前下载过 `bge-small-zh-v1.5`，不需要联网
- **零成本**：两行环境变量，不用改其他代码
- **稳定**：开发阶段完全离线，不受网络影响

### 收获
第三方库的默认行为（自动联网检查）在国内网络环境下需要显式关闭，环境变量是最轻量的方式。

---

## 故事 4：uvicorn --reload 下调 asyncio.run 报错

### 问题
在 FastAPI 模块顶层调用 `asyncio.run(load_tools())` 加载 MCP 工具，报错：
```text
RuntimeError: asyncio.run() cannot be called from a running event loop
```

### 分析
- uvicorn 启动后已经在一个事件循环里
- 模块顶层代码在 import 时执行，此时事件循环正在运行
- `asyncio.run()` 不能嵌套调用

### 方案
改用 FastAPI 的 `lifespan` 生命周期钩子：
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    client = MultiServerMCPClient({...})
    tools = await client.get_tools()
    agent = create_agent(model, tools=tools)
    yield
    print("[关闭] 服务停止")

app = FastAPI(lifespan=lifespan)
```

### 为什么这么选
- **官方推荐**：lifespan 是 FastAPI 替代废弃的 `on_event` 的标准方案
- **异步友好**：在事件循环里执行，可以 `await`
- **生命周期清晰**：yield 之前是启动逻辑，之后是关闭逻辑

### 收获
异步框架下，初始化逻辑应该放在生命周期钩子里，而不是模块顶层。

---

## 故事 5：MCP 工具同步调用报错

### 问题
Agent 调 MCP 工具时，报错：
```text
StructuredTool does not support sync invocation.
```

### 分析
- MCP 工具是异步的（跨进程 stdio 通信）
- 但 Agent 默认用 `agent.invoke()` 和 `agent.stream()`，是同步调用
- 同步方法调异步工具，不兼容

### 方案
改用异步版本：
```python
# 之前
result = agent.invoke({...})
for chunk in agent.stream({...}):

# 之后
result = await agent.ainvoke({...})
async for chunk in agent.astream({...}):
```

### 为什么这么选
- **FastAPI 本身就是异步的**：用异步版本更自然
- **性能更好**：异步不阻塞事件循环，支持更高并发
- **必须的**：MCP 工具只有异步接口，没有同步选项

### 收获
异步工具调用链必须全程异步，不能混用同步和异步。

---

## 故事 6：流式输出吞进中间步骤

### 问题
SSE 流式输出时，用户看到了不该看到的内容：
```text
data: {"text": "I"}
data: {"text": "'ll"}
data: {"text": " look"}
data: {"text": " that"}
data: {"text": " up"}
...
data: {"text": "- MCP 是模型上下文协议..."}   ← 工具返回结果也被吐出来了
```

### 分析
Agent 的执行过程分两段：
1. **工具调用前**：模型输出英文思考（"I'll look that up..."）→ 不该给用户看
2. **工具调用后**：模型生成最终中文回答 → 该给用户看

`astream` 会把所有 token 都吐出来，包括中间步骤。

### 方案
用 `saw_tool` 标志区分两段：
```python
saw_tool = False
pending = []

async for chunk in agent.astream(...):
    if msg_chunk.type == "tool":
        saw_tool = True       # 分界线
        pending = []          # 清空中间步骤
        continue
    if saw_tool:
        yield f"data: {json.dumps({'text': msg_chunk.content})}\n\n"   # 实时输出
    else:
        pending.append(msg_chunk.content)   # 缓存，不输出
```

### 为什么这么选
- **用户体验**：只显示最终回答，不显示内部思考
- **设计权衡**：RAG 场景（有工具调用）完全正常；纯聊天场景（无工具）会先缓存再一次性输出，打字机效果打折，但可接受

### 收获
流式接口需要区分"内部过程"和"用户可见内容"，不能简单地把所有 token 都转发。

---

## 故事 7：Docker 构建慢

### 问题
第一次 `docker compose up --build` 花了 **19 分钟**。

### 分析
Dockerfile 把所有步骤写在一起，每次构建都要重新装依赖、下载模型。

### 方案
利用 Docker 层缓存，把不常变的部分放前面：
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .                              # 先复制依赖清单
RUN pip install -r requirements.txt                  # 装依赖（这层缓存）
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "...SentenceTransformer(...)"          # 下载模型（这层缓存）
COPY mcp_test/mcp_rag_server.py ./mcp_test/          # 再复制代码
COPY rag_api/api_agent.py ./rag_api/
```

### 为什么这么选
- **层缓存**：代码改了不影响依赖层，第二次构建几秒
- **镜像可复现**：模型和依赖都打包进镜像，换机器一致

### 收获
Dockerfile 的指令顺序影响构建效率。把变化频率低的放前面，充分利用层缓存。

---

## 故事 8：MCP Server 里的 print 看不到

### 问题
在 `mcp_rag_server.py` 里加 `print("[RAG Server] 索引构建完成")`，跑起来完全看不到输出。

### 分析
MCP 用 stdio 通信，`stdout` 被协议占用。`print` 默认输出到 `stdout`，被协议管道拦截了。

### 方案
改成输出到 `stderr`：
```python
import sys
print("[RAG Server] 索引构建完成", file=sys.stderr)
```

### 为什么这么选
- **stderr 不被占用**：MCP 只用 stdout 通信，stderr 是自由的
- **改一行就行**：不用重构日志系统

### 收获
使用 stdio 通信的协议，日志必须走 stderr，否则会被协议吞掉。这是理解 MCP 底层机制的一个切入点。

---

## 故事 9：等权 RRF 反而更差

### 问题
做混合检索时，等权 RRF（向量 0.5 / BM25 0.5）的召回率比纯向量检索**还低**：
- 纯向量：100%
- 等权混合：97%

### 分析
失败案例："PydanticOutputParser 怎么用？"
- 期望命中 `my_langchain_notes.md`
- 实际命中 `my_project_notes.md` × 3

**原因**：`my_project_notes.md` 里也提到了 PydanticOutputParser，BM25 看到关键词就把它排前面。RRF 奖励"两路都出现"的文档，BM25 的错误被放大，Top-3 全被 `my_project_notes.md` 垄断。

### 方案
加权 RRF：
```python
vector_weight = 0.7
bm25_weight = 0.3
score += vector_weight / (rrf_k + vector_ranks[content])
score += bm25_weight / (rrf_k + bm25_ranks[content])
```

### 为什么这么选
- **向量检索更可靠**：语义匹配比关键词匹配更少误判
- **权重是超参数**：0.7/0.3 是我调出来的，可以按数据调整
- **不是非黑即白**：不是"用混合检索"，而是"怎么用混合检索"

### 效果
- 等权：97%
- 加权：100%

### 收获
混合检索不一定比纯向量好，关键是权重调优。任何"组合方案"都要评估它的边际收益，而不是盲目叠加。

---

## 故事 10：MMR 是 chunk 级多样性

### 问题
以为 MMR 会强制"Top-3 来自不同文档"，但实际结果里出现了两个 `langchain.md`。

### 分析
MMR 公式：
```
MMR = λ × 相关性 - (1-λ) × 与已选 chunk 的最大相似度
```

**它惩罚的是"和已选 chunk 相似的新 chunk"，不是"和已选 chunk 同文档的新 chunk"。**

所以同一文档里差异大的两个 chunk，MMR 也会都选。

### 方案
接受这个事实，理解 MMR 的边界。如果需要"文档级多样性"，加一层按 source 去重的后处理：
```python
def diverse_by_source(query, k=3):
    results = vectorstore.max_marginal_relevance_search(query, k=k*3, fetch_k=10)
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
```

### 为什么这么选
- **MMR 的设计目标**：内容多样性，不是来源多样性
- **没有对错**：看下游需求——要内容多样用 MMR，要来源多样加去重

### 收获
理解算法的时候要看它的公式和目标，不能凭感觉假设它的行为。MMR 的公式决定了它是 chunk 级多样性。

---

## 故事 11：Agent 可能死循环或 Token 爆炸

### 问题
Agent 在复杂任务下可能**无限调用工具**，或者**单次输出过长**，浪费 Token。

比如用户问一个模糊的问题，模型可能反复检索、反复思考，陷入循环。

### 分析
- LangGraph 的 Agent 本质是循环：调模型 → 调工具 → 调模型 → ...
- 如果模型判断一直"需要调工具"，循环永远不结束
- 单次输出也没有上限，模型可能生成很长的回答

### 方案
两道护栏：

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
```

### 为什么这么选
- **`recursion_limit=10`**：正常 RAG 最多循环 2~3 次（调一次工具 + 生成回答），10 是安全上限
- **`max_tokens=2000`**：覆盖绝大多数回答，异常长回答会被截断
- **异常处理**：护栏触发后抛异常，被 `try/except` 捕获，以 SSE 格式返回友好错误，服务不崩

### 收获
**Agent 是循环结构，必须有终止条件。** 任何循环都要考虑"什么时候停"，否则会有资源耗尽风险。这是生产级 Agent 的基本要求。

---

## 故事 12：多轮对话下的缓存串台

### 问题
加了多轮对话记忆后，Redis 缓存的 key 还是只按 query 的 MD5：

```python
key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
```

**结果**：会话 A 问"它是什么"拿到答案 X，缓存；会话 B 问"它是什么"直接返回 X，但会话 B 里的"它"可能是完全不同的东西。

### 分析
- 加多轮对话前：同一问题答案永远一样，缓存 key 只按 query 没问题
- 加多轮对话后：同一问题在**不同上下文**里答案不同
- 缓存 key 不带 thread_id → 会话之间**串台**

### 方案
缓存 key 带上 thread_id：

```python
def get_cached(thread_id: str, query: str):
    key = f"chat:{thread_id}:{hashlib.md5(query.encode()).hexdigest()}"
    return r.get(key)

def set_cache(thread_id: str, query: str, answer: str, ttl: int = 3600):
    key = f"chat:{thread_id}:{hashlib.md5(query.encode()).hexdigest()}"
    r.set(key, answer, ex=ttl)
```

### 为什么这么选
- **按会话隔离**：每个用户的缓存互不影响
- **仍保留 MD5**：避免中文/特殊字符问题，固定长度
- **代价**：缓存命中率下降（因为按会话分桶），但正确性优先

### 收获
**加新功能时，要检查它是否影响已有设计的假设。** 加多轮对话前，"同一 query 同一答案"是成立的；加了之后这个假设不成立，缓存设计就要跟着改。这是**功能演进引发的设计回归**。

---

## 故事 13：InMemorySaver 的局限

### 问题
用 `InMemorySaver` 做多轮对话记忆，功能正常，但有个根本问题：**服务重启后，所有会话记忆丢失。**

### 分析
- `InMemorySaver` 把记忆存在**进程内存**里
- 服务重启 → 内存清空 → 用户的历史对话全没了
- 用户回到会话里问"它是什么"，Agent 完全不知道上下文

### 方案
**开发测试**：用 `InMemorySaver`，简单够用。

**生产环境**：换持久化的 Checkpointer：

| 方案 | 特点 |
|---|---|
| `RedisSaver` | 存 Redis，快，适合已有 Redis 的项目 |
| `PostgresSaver` | 存 PostgreSQL，官方推荐，功能全 |
| `SQLiteSaver` | 存本地文件，轻量，适合单机 |

改动很小，只需换 `checkpointer` 参数：

```python
# 开发
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()

# 生产
from langgraph.checkpoint.redis import RedisSaver
memory = RedisSaver(redis_url="redis://localhost:6379")

# 用法完全一样
agent = create_agent(model, tools=tools, checkpointer=memory, ...)
```

### 为什么这么选
- **开发用 InMemorySaver**：零配置，跑起来就行
- **生产用 RedisSaver**：项目已有 Redis，复用基础设施
- **PostgresSaver 备选**：如果项目用 PG 且需要更强的持久化保证

### 收获
**开发和生产的技术选型经常不同。** 开发追求"跑得快、配得少"；生产追求"稳、持久、可扩展"。面试时能说清"我用 InMemorySaver 开发，生产会换 RedisSaver"，比只会用一个方案强得多。

---

## 故事 14：检索不到内容怎么避免幻觉

### 问题
RAG 系统在知识库中找不到相关内容时，如果直接把不相关的片段塞给 LLM，模型可能基于这些片段产生幻觉，编造听起来合理的答案。

### 分析
- LLM 有"必须回答"的倾向，即使输入不相关
- Prompt 里写"只使用资料中的信息"能缓解，但不能完全避免
- **根本解法**：检索层先判断"有没有相关内容"，没有就不调 LLM

### 方案
在 `hybrid_search` 里加距离阈值：

```python
MAX_DISTANCE = 1.1

def hybrid_search(query, k=3, max_distance=MAX_DISTANCE):
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k*3)
    best_distance = results_with_scores[0][1]

    if best_distance > max_distance:
        return []   # 降级

    # 正常 RRF 合并
    ...
```

`search_knowledge` 工具处理空结果：

```python
if not results:
    return "知识库中没有找到相关信息。"
```

### 阈值怎么定（实测数据）

| 类别 | 距离范围 |
|---|---|
| 正常问题 | 0.58 ~ 0.89 |
| 乱码 | 1.15 |
| 无关问题 | 1.27 ~ 1.34 |

**选 1.1**：正常问题全过，无关问题全拦。

### 为什么用向量距离，不用 RRF 分数
- RRF 分数范围窄（0.005~0.02），无法区分好坏
- 向量 L2 距离直接反映语义相似度，范围 0~2，好判断

### 收获
**降级策略比"更好的 Prompt"更可靠。** Prompt 是软约束，模型可能不遵守；降级是硬约束，在代码层就拦住了。生产级 RAG 必须有两层防护：Prompt 防幻觉 + 检索层降级。

---

## 故事 15：RAG 评估不能只看召回率

### 问题
一开始评估 RAG 只用召回率。但召回率有个问题：**只要期望文档出现在 Top-3 里就算命中**，不管它排第 1 还是第 3，也不管 Top-3 里其他两条是不是相关。

### 分析
单看召回率会漏掉这些信息：
- **精确率**：Top-K 里有几条是真正相关的
- **MRR**：第一个正确结果排多前
- **多样性**：结果覆盖了多少不同来源

### 方案
加三个指标：

```python
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
    print(f"召回率：{hit/n:.0%}")
    print(f"精确率：{precision_sum/n:.2f}")
    print(f"MRR：{mrr_sum/n:.2f}")
```

### 实测结果

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 相似度 | 100% | **0.87** | 0.96 | 1.37 |
| MMR | 100% | 0.83 | **0.97** | 1.43 |
| MMR + 去重 | 100% | 0.77 | **0.97** | **1.57** |

### 关键发现
- **精确率和多样性呈 trade-off**：想要多角度信息，就要接受部分结果不直接相关
- **MRR 几乎不变**：三种策略的 Top-1 都很准
- **选策略看业务**：追求精确用相似度，需要多角度用 MMR+去重

### 收获
**单一指标会误导。** 只看召回率以为三种策略一样好，加上精确率和多样性才看出它们的 trade-off。**评估 RAG 要多个指标一起看。**

---

## 故事 16：InMemorySaver 重启丢记忆

### 问题
用 `InMemorySaver` 做多轮对话，功能正常。但服务重启后，用户的历史对话全没了。

### 分析
- `InMemorySaver` 把记忆存在**进程内存**里
- 服务重启 → 内存清空 → 用户的历史对话全没了
- 用户回来问"它是什么"，Agent 完全不知道上下文

### 方案
**两层记忆**：

| 层 | 存储 | 作用 |
|---|---|---|
| 短期 | `InMemorySaver` | 单次会话内多轮，快 |
| 长期 | MySQL | 跨会话持久化，不怕重启 |

每次请求，先从 MySQL 加载最近 N 轮历史，拼进 messages：

```python
HISTORY_ROUNDS = 3

def load_recent_history(thread_id, rounds=HISTORY_ROUNDS):
    conn = pymysql.connect(...)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM conversations "
                "WHERE thread_id = %s ORDER BY created_at DESC LIMIT %s",
                (thread_id, rounds * 2)
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [{"role": r, "content": c} for r, c in reversed(rows)]

# 在接口里
history = load_recent_history(thread_id)
messages = history + [{"role": "user", "content": payload.message}]
result = await agent.ainvoke({"messages": messages}, config=config)
```

### 关键设计
- **限制轮数**（3 轮）：避免上下文爆炸
- **正序返回**：模型看到的时间线要正确
- **每次实时加载**：MySQL 是权威数据源

### 验证
1. 第一轮问 MMR → `[长期记忆] 加载了 0 条历史消息`
2. **重启服务**
3. 第二轮问"它和普通相似度有什么区别？" → `[长期记忆] 加载了 2 条历史消息` + 能理解"它" ✅

### 收获
**开发用 InMemorySaver，生产用持久化方案。** 不是替代关系，是互补关系。InMemorySaver 快但易失，MySQL 慢但持久。生产系统往往两者都用：MySQL 存历史，InMemorySaver 做单次会话加速。

---

## 总结

这些坑覆盖了 Agent 开发中常见的几类问题：

| 类别 | 核心教训 |
|---|---|
| 模型 API 差异 | 用适配层屏蔽，不能假设所有模型都一样 |
| 依赖冲突 | 虚拟环境隔离，冲突时降级成本最低 |
| 网络问题 | 第三方库默认行为要显式关闭 |
| 异步编程 | 异步框架下初始化放 lifespan，工具链全程异步 |
| 用户体验 | 流式输出要区分内部过程和用户可见内容 |
| 构建优化 | Dockerfile 指令顺序影响缓存效率 |
| 协议理解 | stdio 通信要用 stderr 打日志 |
| 检索算法 | 混合检索要调权重，MMR 是 chunk 级多样性 |
| 生产级意识 | Agent 要有护栏，记忆要持久化 |
| 功能演进回归 | 加新功能要检查旧设计的假设是否还成立 |
| 幻觉治理 | Prompt 软约束 + 检索层硬降级，两层防护 |
| 评估方法 | 单一指标会误导，要多个指标一起看 |
| 记忆持久化 | 短期 InMemorySaver + 长期 MySQL，两层互补 |

**每个问题的解决过程都体现了同一个原则：先分析根本原因，再选择方案，最后量化效果。**