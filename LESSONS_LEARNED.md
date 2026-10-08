# 项目踩坑与复盘

> 记录项目开发过程中遇到的问题、分析过程和解决方案。
> 每个故事按"问题 → 分析 → 方案 → 收获"结构。
> 最后更新：2026-10-08（新增 Docker 部署系列 7 个故事）

## 快速索引

| 类别 | 对应故事 |
|---|---|
| 模型 API 限制 | 故事 1 |
| 依赖冲突 | 故事 2 |
| 网络问题 | 故事 3 |
| 异步编程 | 故事 4、5 |
| 用户体验 | 故事 6 |
| 构建优化 | 故事 7、26 |
| 协议理解 | 故事 8 |
| 检索算法 | 故事 9、10 |
| 生产级意识 | 故事 11 |
| 功能演进回归 | 故事 12 |
| 幻觉治理 | 故事 14 |
| 评估方法 | 故事 15 |
| 记忆持久化 | 故事 13、16 |
| 架构重构 | 故事 17 |
| 安全防护 | 故事 18 |
| 缓存正确性 | 故事 19 |
| 依赖管理 | 故事 20 |
| **磁盘与 Docker 环境** | **故事 21、22** |
| **依赖下载与平台兼容** | **故事 23、24** |
| **跨平台 wheel 陷阱** | **故事 25** |
| **模型缓存与容器** | **故事 26** |
| **checkpoints 持久化** | **故事 27** |

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

### 已知不足
如果模型在工具调用后不再输出文本（比如工具结果直接就是答案），`started` 永远是 False，只发 `[DONE]`，用户看到空白。**加兜底**：`if not started and not pending: yield fallback`。

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

### 更新（2026-10-08）
本次构建又踩了新坑——把模型下载放在构建期会导致网络不稳定时构建失败。**最终方案**：模型下载移出构建期，改由运行时 volume 挂载 `hf_cache`。详见故事 26。

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

### 已知不足
30 条样本上，97% 和 100% 差 1 条，可能只是噪声。需要扩充样本 + 加随机基线。

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

### 已知不足
见故事 19。

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
| `AsyncSqliteSaver` | 存本地 SQLite 文件，轻量，适合单机 |
| `AsyncPostgresSaver` | 存 PostgreSQL，官方推荐，多实例共享 |
| `AsyncRedisSaver` | 存 Redis，快，适合已有 Redis 的项目 |

改动很小，只需换 `checkpointer` 和导入：

```python
# 开发
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()

# 单机部署
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
    agent = create_agent(model, tools=tools, checkpointer=saver, ...)

# 生产
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string("postgresql://...") as saver:
    agent = create_agent(model, tools=tools, checkpointer=saver, ...)
```

**接口完全一致**，业务代码不用改。

### 为什么这么选
- **开发用 InMemorySaver**：零配置，跑起来就行
- **单机部署用 AsyncSqliteSaver**：一个文件搞定，重启不丢
- **生产用 AsyncPostgresSaver**：多实例共享，高可用

### 收获
**开发和生产的技术选型经常不同。** 开发追求"跑得快、配得少"；生产追求"稳、持久、可扩展"。面试时能说清"我用 InMemorySaver 开发，单机用 AsyncSqliteSaver，生产会换 AsyncPostgresSaver"，比只会用一个方案强得多。

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

### 已知不足
`MAX_DISTANCE = 1.1` 的校准实验不够系统。应该对正样本和负样本分别统计 best_distance 分布，取分离点，画出 PR 曲线。

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

### 已知不足
- **文档级召回而非 chunk 级**：用 `source`（文件名）判断命中，只有两三篇文档，命中太容易，含金量不高。生产环境应该标注 chunk id 或关键句。
- **30 条样本太少**：3% 的差异（97% vs 93%）是一条样本的差别，统计上无意义。应该扩到 200 条以上。
- **没有端到端指标**：检索对了不等于回答对了。应该加 LLM-as-judge 或人工标注的 answer correctness。
- **没有随机基线**：补测随机检索器，30 条上召回率 84.1%，说明当前指标判别力只有 15.9 个点。

---

## 故事 16：InMemorySaver 重启丢记忆

### 问题
用 `InMemorySaver` 做多轮对话，功能正常。但服务重启后，用户的历史对话全没了。

### 分析
- `InMemorySaver` 把记忆存在**进程内存**里
- 服务重启 → 内存清空 → 用户的历史对话全没了
- 用户回来问"它是什么"，Agent 完全不知道上下文

### 方案
把 `InMemorySaver` 换成 **`AsyncSqliteSaver`**，让 LangGraph 的 checkpointer 直接做持久化。

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        agent = create_agent(
            model, tools=tools,
            checkpointer=saver,   # 持久化 saver
            system_prompt="..."
        )
        yield
```

调用时只传当前问题，**不手动拼历史**：

```python
config = {"configurable": {"thread_id": thread_id}}
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": payload.message}]},
    config=config
)
```

### 关键设计
- **checkpointer 管历史**：Agent 自动从 SQLite 按 thread_id 加载
- **只传当前一轮**：不手动拼 `history + [当前问题]`
- **`async with` 包裹**：保证服务停止时正确关闭 SQLite 连接

### 验证
1. 第一轮问"MMR 是什么？"
2. **重启服务**
3. 第二轮问"它和普通相似度有什么区别？" → 仍能理解"它" ✅

### 收获
**让框架做框架的事。** LangGraph 的 checkpointer 就是设计来管状态的，不要在外面手动糊一层。生产环境可以无缝升级到 `AsyncPostgresSaver`，业务代码完全不用改。

### 更新（2026-10-08）
上面的验证只覆盖了**进程重启**，没覆盖**容器重建**。`docker compose down` 后容器销毁，`checkpoints.db` 也丢了。详见故事 27。

---

## 故事 17：双源记忆打架（架构重构）

### 问题
在故事 16 之前，我用的是"两层记忆"方案：短期 `InMemorySaver` + 长期 MySQL 手动加载。看起来很美，但**两个源会打架**。

### 分析
代码里同时有：

```python
memory = InMemorySaver()                          # 源 1：LangGraph checkpointer
history = load_recent_history(thread_id)          # 源 2：MySQL 手动加载
messages = history + [{"role": "user", ...}]      # 源 3：手动拼 messages
result = await agent.ainvoke({"messages": messages}, config=config)
```

**三个具体问题**：

**问题 A：消息格式混用**
- `InMemorySaver` 存的是 LangChain 的 `HumanMessage`/`AIMessage` 对象
- 手动拼的是 OpenAI 的 `{"role": "user"}` dict
- 两种格式在 state 里混着

**问题 B：历史重复累积**
- `InMemorySaver` 已经在按 `thread_id` 累积 state
- 传入的 messages 会再叠加一遍（取决于 reducer 实现）

**问题 C：消息对不完整**
- `LIMIT rounds*2` 可能把 `AIMessage(tool_calls=...)` 和对应的 `ToolMessage` 截断
- OpenAI 协议要求 `tool_calls` 和 `tool_call_id` 必须成对出现
- 这种 state 在下次请求时可能直接触发 400

### 方案
**统一到 LangGraph 的 checkpointer 做唯一真相源**，MySQL 降级为展示层。

| 角色 | 之前 | 之后 |
|---|---|---|
| 记忆源 | InMemorySaver + MySQL 手动拼 | **AsyncSqliteSaver（唯一真相源）** |
| MySQL | 记忆源 | **展示层**（给 /history 用） |
| 消息格式 | OpenAI dict 手动拼 | 统一交给 LangGraph |
| 手动加载历史 | ✅ 有 | ❌ 没有 |

**代码改动**：
1. `InMemorySaver` → `AsyncSqliteSaver`
2. 删掉 `load_recent_history` 函数
3. `/chat` 和 `/chat/stream` 只传 `{"messages": [{"role": "user", ...}]}`

### 为什么这么选
- **单一真相源**：一份历史只由一个组件管，不打架
- **符合框架意图**：LangGraph 的 checkpointer 就是干这个的
- **可平滑升级**：Sqlite → Postgres → Redis，接口不变

### 收获
**架构问题比功能 bug 更值得重视。** "两个源同时管历史"看起来能用，但埋着消息格式混用、历史重复、协议违规等隐患。**一旦发现职责重叠，就要立即统一到单一真相源。**

这也印证了一个原则：**让框架做框架的事。** 不要因为"我能自己写"就在外面糊一层，那只会让架构越来越乱。

---

## 故事 18：错误信息泄漏（安全防护）

### 问题
最初在流式接口的异常处理里，直接把异常字符串返回给客户端：

```python
except Exception as e:
    yield f"data: {json.dumps({'error': f'服务出错了：{str(e)}'})}\n\n"
```

**这有安全隐患**：`str(e)` 可能包含 API key、文件路径、SQL 结构、内部 IP 等敏感信息。

### 分析
- 异常堆栈通常包含**上游服务返回的原文**，比如 `Authentication Fails, Your api key: ****0cb1 is invalid`
- 如果直接给客户端，攻击者能通过报错探测系统结构
- 比如 `Connection refused to 10.0.1.5:3306` 暴露内网 IP
- 比如 SQL 语法错误暴露表结构和字段名

### 方案（初版）
**服务端记录完整异常，客户端只看到友好提示。**

```python
def friendly_error(e: Exception) -> str:
    msg = str(e).lower()
    if "authentication" in msg or "401" in msg or "api key" in msg:
        return "模型服务认证失败，请联系管理员"
    if "timeout" in msg or "timed out" in msg:
        return "请求超时，请稍后再试"
    if "rate" in msg or "429" in msg:
        return "请求过于频繁，请稍后再试"
    if "connection" in msg or "connect" in msg:
        return "服务暂时不可用，请稍后再试"
    return "服务暂时不可用，请稍后再试"
```

### 新问题：子串匹配的误判
这个初版用**子串匹配**判断异常类型，但 `"rate" in msg` 会误匹配：

| 原始异常 | 被误判为 |
|---|---|
| `failed to generate response` | 请求过于频繁（因为 `generate` 里有 `rate`） |
| `cannot operate on closed file` | 请求过于频繁（因为 `operate` 里有 `rate`） |
| `separate connection failed` | 请求过于频繁（因为 `separate` 里有 `rate`） |

**根本问题**：用自然语言子串匹配，边界情况多，不可靠。

### 方案（修正版）
**用异常类型 + HTTP 状态码判断，不做自然语言子串匹配。**

```python
def friendly_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)  # openai 异常带这个
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

在接口里：

```python
try:
    ...
except Exception as e:
    # 服务端记录完整异常（含堆栈），方便排查
    print(f"[错误] {type(e).__name__}: {str(e)}")
    # 客户端只看到友好提示
    raise HTTPException(status_code=500, detail=friendly_error(e))
```

### 为什么这么选
- **安全**：不泄漏 API key、内网 IP、SQL 结构等
- **准确**：用异常类型 + 状态码，不会误判
- **可排查**：服务端日志保留完整信息，运维能定位问题

### 收获
**生产环境的异常处理必须分层**：
- **服务端日志**：完整异常 + 堆栈，方便排查
- **客户端响应**：映射后的友好提示，不泄漏内部信息

**更重要的教训**：用**结构化判断**（异常类型、状态码）而不是**自然语言匹配**（子串）。自然语言的边界情况太多，总有误判。

---

## 故事 19：缓存 key 不含上下文（缓存正确性）

### 问题
加了多轮对话后，缓存 key 从"只按 query"改成"按 thread_id + query"：

```python
key = f"chat:{thread_id}:{hashlib.md5(query.encode()).hexdigest()}"
```

看起来解决了会话隔离问题，但**仍有一个正确性 bug**：

**场景**：
```text
第 1 轮：用户问 "MMR 是什么？" → 答案 A → 缓存 key = chat:user1:abc123
第 2 轮：用户问 "它和普通相似度有什么区别？" → 答案 B
第 3 轮：用户又问 "MMR 是什么？"
        → 命中缓存 key chat:user1:abc123 → 返回答案 A
        → 但此时上下文变了（多了第 2 轮），答案应该不一样
```

**根因**：上下文没进 key，同一个 query 在不同上下文里命中同一缓存。

### 分析
- 多轮对话下，**答案依赖上下文**，不只是 query
- 上下文变了（新增了对话历史），同样的 query 应该重新生成
- 只按 query 做 key，等于假设"同一问题答案永远一样"，这个假设在**多轮对话**下不成立
- 这和故事 12（串台问题）是**同一类问题的不同表现**：
  - 故事 12：跨会话串台，靠 `thread_id` 解决
  - 故事 19：同会话上下文错配，靠"上下文版本号"解决

### 方案（初版）
把**上下文版本号**也加进 key：

```python
def get_context_version(thread_id: str) -> int:
    """获取该会话的上下文版本号（= MySQL 里的消息数）"""
    conn = pymysql.connect(...)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM conversations WHERE thread_id = %s",
                (thread_id,)
            )
            count = cur.fetchone()[0]
    finally:
        conn.close()
    return count

def get_cached(thread_id: str, query: str, context_version: int):
    key = f"chat:{thread_id}:v{context_version}:{hashlib.md5(query.encode()).hexdigest()}"
    return r.get(key)

def set_cache(thread_id: str, query: str, answer: str, context_version: int, ttl: int = 3600):
    key = f"chat:{thread_id}:v{context_version}:{hashlib.md5(query.encode()).hexdigest()}"
    r.set(key, answer, ex=ttl)
```

**缓存 key 四段结构**：

| 段 | 作用 |
|---|---|
| `chat:` | 命名空间隔离 |
| `thread_id` | 防止跨会话串台 |
| `v{context_version}` | 防止同会话上下文错配 |
| `{md5}` | query 的 MD5 |

### 实测验证

```text
第一次：[缓存写入] MMR 是什么？ (v0)
第二次：[缓存写入] MMR 是什么？ (v2)   ← 版本变了，缓存没命中
```

第二次问同一个问题，因为 `context_version` 从 `v0` 变成 `v2`，**缓存 key 不一样，所以没命中，重新调了 Agent**。

### 新问题：命中率结构性为 0
看起来"正确性优先"，但**实际是缓存彻底失效**：

- `context_version` 来自 MySQL 行数
- 每轮对话往 MySQL 写 2 行（human + ai）
- 所以版本号每轮 **+2**
- 同一个问题问第二次，key 一定不一样，**必然 miss**

**结论**：这个缓存的命中率结构性为 0。除了增加两次 MySQL 查询开销，没有任何收益。

### 最终方案：删掉缓存
既然命中率为 0，缓存就是纯负优化。**直接删掉。**

```python
# 删掉的代码：
# import redis
# import hashlib
# r = redis.Redis(...)
# def get_context_version(...)
# def get_cached(...)
# def set_cache(...)

# 简化为：
@app.post("/chat")
async def chat(payload: ChatRequest):
    thread_id = payload.thread_id

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    try:
        result = await agent.ainvoke({...}, config=config)
        answer = result["messages"][-1].content
    except Exception as e:
        raise HTTPException(500, detail=friendly_error(e))

    try:
        save_conversation(thread_id, "human", payload.message)
        save_conversation(thread_id, "ai", answer)
    except Exception as e:
        print(f"[MySQL 写入失败] {type(e).__name__}: {str(e)}")

    return ChatResponse(reply=answer)
```

### 为什么最终选择删缓存
- **命中率为 0**：加了个寂寞
- **破坏双源一致性**：缓存命中时写 MySQL，但 agent 的 checkpointer 没走
- **负优化**：增加 2 次 MySQL 查询开销
- **正确性 bug**：如果 context_version 算错，会返回错误答案

**删掉比修更简单、更干净。**

### 收获
**缓存设计要跟着数据模型演进。** 从单轮对话到多轮对话，缓存的假设从"query → 答案"变成"query + context → 答案"。**假设变了，key 的结构就要跟着改。**

**但更重要的是**：**缓存不是免费午餐。** 加缓存前先算账：
- 命中率是多少？
- 每次 miss 的代价 vs 每次命中的收益？
- 缓存维护成本（key 设计、失效策略、一致性）？

**如果命中率结构性为 0，缓存就是负优化。删掉是最优解。**

这也是**功能演进引发设计回归**的又一个例子：
- 加多轮对话 → 故事 12 发现串台 → 加 thread_id
- 加多轮对话 → 故事 19 发现上下文错配 → 加 context_version
- 加 context_version → 发现命中率 0 → **删掉整个缓存**

**每次加功能，都要回头检查一遍所有假设，包括"加这个功能到底值不值"。**

---

## 故事 20：requirements.txt 从未在干净环境验证

### 问题
项目里 `pip install` 装了一堆包，但 `requirements.txt` 只写了最初那 10 个。

在干净 venv 里跑 `pip install -r requirements.txt` 后，`python -c "import rag_api.api_agent"` 立刻报错：
```text
ModuleNotFoundError: No module named 'redis'
```

### 分析
代码里实际 import 的第三方包有 18 个，`requirements.txt` 里只声明了 10 个。**缺 8 个**：

| 缺失包 | 被谁 import |
|---|---|
| `redis` | api_agent.py |
| `pymysql` | api_agent.py |
| `langchain-community` | mcp_rag_server.py（FAISS） |
| `langchain-text-splitters` | mcp_rag_server.py（切分） |
| `langgraph-checkpoint-sqlite` | api_agent.py（AsyncSqliteSaver） |
| `aiosqlite` | AsyncSqliteSaver 的底层依赖 |
| `jieba` | mcp_rag_server.py（中文分词） |
| `rank-bm25` | mcp_rag_server.py（BM25） |

**根因**：开发过程中逐个 `pip install` 补包，**忘了同步回 `requirements.txt`**。

### 杀伤力
- 面试官 clone 下来，`pip install -r requirements.txt` 装到一半就报错
- Docker 构建时，容器里缺 `redis`，`uvicorn` 一 import 就崩
- CI 流水线同理

**这是"项目能不能在别人机器上跑"的最底线问题。**

### 方案
1. 遍历项目里所有 `.py` 文件，提取所有 `import` 语句
2. 映射到对应的 pip 包名（注意：import 名里的下划线 `_` 通常要换成连字符 `-`）
3. 补全 `requirements.txt`
4. **在干净 venv 里验证**：
   ```cmd
   python -m venv .venv-test
   .venv-test\Scripts\activate
   pip install -r requirements.txt
   python -c "import rag_api.api_agent; print('导入成功')"
   ```

### 修复后的清单（18 个）

```text
fastapi>=0.136.0
uvicorn>=0.48.0
langchain>=1.4.0
langchain-openai>=1.6.0
langchain-mcp-adapters>=0.3.0
langchain-community>=0.4.0
langchain-text-splitters>=1.0.0
langgraph>=1.2.0
langgraph-checkpoint-sqlite>=2.0.0
aiosqlite>=0.20.0
fastmcp>=3.4.0,<4.0.0
sentence-transformers>=5.5.0
faiss-cpu>=1.14.0
python-dotenv>=1.2.0
pymysql>=1.1.0
jieba>=0.42.0
rank-bm25>=0.2.2
```

（`redis` 暂时保留，预留做限流）

### 为什么这个问题容易被忽略
- 开发时，**你机器上碰巧有**这些包（之前手动装过）
- 所以你本地能跑，感觉不到问题
- 但换机器就崩

**依赖清单必须经过干净环境验证，否则就是装饰品。**

### 收获
**每装一个新包，立刻 `pip freeze` 或手动补进 `requirements.txt`。** 更好的做法是用 `uv` 或 `poetry`，它们在装包时自动维护依赖清单，不需要手动同步。

**面试时能讲**：
> "我之前踩过一个坑：项目里 `pip install` 装了一堆包，但 `requirements.txt` 只写了最初那 10 个。后来在一个干净 venv 里跑 `pip install -r requirements.txt`，立刻发现 `redis`、`pymysql`、`jieba`、`rank-bm25` 这些都没声明。修法是把所有 `.py` 文件的 import 语句扫一遍，映射到 pip 包名，补全清单，再在新环境里验证。这件事让我意识到：依赖清单必须经过干净环境验证，否则就是装饰品。"

---

## 故事 21：Docker 构建报 `unpigz corrupted`（2026-10-08）

### 问题
Docker 构建到 `exporting to image` / `unpacking to image` 阶段突然失败：

```text
#16 exporting to image
#16 unpacking to docker.io/library/agent-agent-api:latest
#16 ERROR: failed to extract layer sha256:f7dfae7ab81225ebfe4b866133d90a08b596073e07813dcada89a0823070199d: 
    exit status 1: unpigz: skipping: <stdin>: corrupted -- incomplete deflate data
failed to solve: Unavailable: error reading from server: EOF
```

### 分析
- 报错位置在 `unpacking`（写盘阶段），不是编译阶段 → **不是代码问题，是环境问题**
- 报错关键词 `corrupted -- incomplete deflate data` → 写入被截断
- 第一反应查磁盘：
  ```powershell
  Get-PSDrive C
  # 结果：C 盘 200 GB，Free 0.00 GB ← 命中
  wsl -d docker-desktop df -h
  # 结果：/mnt/host/c 200.0G 200.0G 0 100% ← 确认
  ```
- Docker 解压镜像层时要往 C 盘写数据，**C 盘一个字节都写不进去**，管道被截断

### 方案
1. **清 C 盘**（管理员 PowerShell）：
   ```powershell
   cleanmgr /sageset:1    # 弹出窗口，全勾
   cleanmgr /sagerun:1

   Remove-Item "C:\Windows\SoftwareDistribution\Download\*" -Recurse -Force
   Remove-Item "$env:TEMP\*" -Recurse -Force
   Remove-Item "$env:USERPROFILE\.cache\huggingface" -Recurse -Force
   pip cache purge
   ```
   清出约 10 GB。

2. **把 Docker 数据迁到 D 盘**（见故事 22）。

### 为什么是"磁盘满"而不是"改代码"
`unpigz corrupted` 和 `incomplete deflate data` 这种错误，**永远是 I/O 层面的问题**，不是程序 bug。看到这类错误，排查顺序应该是：
1. 磁盘空间
2. 磁盘 I/O 故障（坏道、文件系统损坏）
3. 网络传输中断（如果在 pull 阶段）
4. Docker 缓存损坏

**先查磁盘，再查别的。**

### 收获
**Docker 构建失败的第一排查项是磁盘空间。** 尤其在中国 Windows 环境下，C 盘经常被一堆缓存、Docker 虚拟磁盘、HuggingFace 模型吃满。**磁盘满时，各种"看似离奇"的构建报错都可能是写盘失败导致的。**

---

## 故事 22：Docker 数据从 C 盘迁到 D 盘（2026-10-08）

### 问题
故事 21 里，清出 10 GB 后 C 盘还是不够——因为 **Docker 的 WSL2 虚拟磁盘 (`ext4.vhdx`) 默认存在 C 盘**，占了 **55.7 GB**。每次构建镜像都会往这个 vhdx 里写，越写越大。

### 分析
- Docker Desktop 在 Windows 上用 WSL2 后端
- WSL2 的虚拟磁盘默认在 `C:\Users\<user>\AppData\Local\Docker\wsl\`
- 这个 vhdx 是**动态扩容**的，但**不会自动缩容**——删了的镜像不会立即释放空间
- 加上 Docker 构建缓存，很容易撑到几十 GB

### 方案
**Docker Desktop 内置了 GUI 迁移功能**：

1. 托盘图标 → Docker Desktop → **Settings** → **Resources** → **Advanced**
2. 找到 **Disk image location**
3. 改成 `D:\Docker`（提前手动建好目录）
4. 点 **Apply & Restart**
5. 等它自动迁移 vhdx（几分钟到十几分钟）

**结果**：
```
C 盘：10.20 GB → 65.25 GB
D:\Docker：55.73 GB
```

### 为什么不用其他方法
- **`wsl --export` + `wsl --import`**：能迁，但会丢 Docker 的元数据，镜像/容器可能全没
- **`Optimize-VHD`**：Windows Home 版没有 Hyper-V 模块，跑不了
- **`docker system prune -a`**：只清镜像和缓存，不动 vhdx 本身
- **GUI 迁移**：官方路径，保留数据，不动 Docker 状态

### 顺手：项目也搬到 D 盘
```cmd
Move-Item "C:\Users\wzyls\Desktop\Agent" "D:\Projects\Agent"
```
失败（被 Docker 进程占用）时用 `robocopy`：
```cmd
robocopy "C:\Users\wzyls\Desktop\Agent" "D:\Projects\Agent" /E /MOVE /R:0 /W:0
```

### 收获
**Docker 数据默认在 C 盘，长期使用会撑爆系统盘。** 装机时第一件事就是改 Disk image location 到其他盘。C 盘紧张时，这个操作能一步腾出几十 GB。

**面试时能讲**：
> "我遇到过 Docker 构建报 `unpigz corrupted`，一开始以为是网络问题，后来查 C 盘发现已经 0 字节。根因是 Docker 的 WSL2 虚拟磁盘默认在 C 盘，占了 55 GB。解决方案是在 Docker Desktop 的 Settings 里改 Disk image location 到 D 盘，一键迁移，C 盘从 10 GB 恢复到 65 GB。"

---

## 故事 23：pip 清华源突然超时（2026-10-08）

### 问题
改完 Dockerfile 重新构建，pip 下载突然开始超时：

```text
ReadTimeoutError: HTTPSConnectionPool(host='pypi.tuna.tsinghua.edu.cn', port=443): 
Read timed out.
```

**但"之前明明不超时"** —— 之前同样用清华源，构建成功过。

### 分析
分两层看：

**第一层：为什么"之前不超时"？**
- 之前构建的镜像里，`RUN pip install` 这一层的 hash 命中缓存
- Docker 看到这一层和上次完全一样，直接 `CACHED`
- **pip 根本没走网络**，所以不会超时

**第二层：为什么"这次超时"？**
- 我改 Dockerfile 时把 pip 命令从：
  ```
  pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
  改成：
  ```
  pip install --no-cache-dir --prefix=/install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- **多了 `--prefix=/install` 五个字**，Docker 认为"命令变了"，缓存失效
- 必须从零下载所有依赖，包括 **CUDA 全家桶 3~5 GB**
- 清华源在下载超大包（triton 248 MB、cudnn 500 MB）时容易超时

### 方案
1. **换阿里云源**（比清华稳定）：
   ```
   -i https://mirrors.aliyun.com/pypi/simple/
   --trusted-host mirrors.aliyun.com
   ```
2. **加超时和重试参数**：
   ```
   --timeout 300
   --retries 10
   ```
3. **根本解法**：用 CPU 版 torch（见故事 24），下载量从 5 GB 砍到 600 MB

### 收获
**"为什么之前不超时？"要分两层思考**：
- 表层：网络抖动
- 深层：Docker 层缓存失效，让之前"不联网"的操作变成了"联网"

**改 Dockerfile 里任何命令的细微参数，都可能导致整层缓存失效。** 改之前想清楚：这一层是不是高频变动？能不能拆出来？

**另一个教训**：不要默认"之前能用现在也能用"。**"之前能用"往往只是因为缓存命中了。**

---

## 故事 24：CPU 版 torch vs CUDA 版（2026-10-08）

### 问题
用默认的 `pip install torch` 会拉**CUDA 全家桶**：

| 包 | 大小 |
|---|---|
| torch | ~900 MB |
| triton | 248 MB |
| nvidia-cudnn-cu13 | ~500 MB |
| nvidia-nccl-cu13 | ~200 MB |
| nvidia-nvshmem-cu13 | ~100 MB |
| nvidia-cusparselt-cu13 | ~100 MB |
| **合计** | **3~5 GB** |

**下载 5 GB 在镜像源上很容易超时。**

### 分析
- 本项目只用 **bge-small-zh-v1.5**（24 MB 小模型）做 embedding
- RAG 检索走 FAISS（CPU）
- LLM 推理走 DeepSeek API（云端，不在容器里跑）
- **容器里唯一用 torch 的地方就是 embedding，CPU 毫秒级**
- **装 CUDA 全家桶是杀鸡用牛刀**

### 方案
在 `requirements.txt` 顶部加两行：
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

**原理**：PyTorch 官方 CPU 源里 torch 版本是 `2.x.x+cpu`。按 PEP 440 版本规则，`2.9.0+cpu > 2.9.0`，**pip 会自动选 CPU 版**。

### 效果对比

| 项目 | CUDA 版 | CPU 版 |
|---|---|---|
| torch | 900 MB + 一堆 CUDA 依赖 | **126 MB** |
| triton | 248 MB | **不装** |
| nvidia-cudnn-cu13 | ~500 MB | **不装** |
| nvidia-nccl-cu13 | ~200 MB | **不装** |
| nvidia-nvshmem-cu13 | ~100 MB | **不装** |
| **总下载量** | **3~5 GB** | **~600 MB** |
| **镜像大小** | 3.51 GB | **2.42 GB** |

### 为什么这么选
- **性能需求**：本项目用不到 GPU，CPU 版完全够用
- **下载稳定**：600 MB 在 300 秒超时内稳定完成
- **镜像小**：少了 1.1 GB
- **构建快**：从 33 分钟降到 5~10 分钟

### 收获
**装依赖前先问："这个项目真的需要 GPU 吗？"** 很多 RAG / Agent 项目只是用 embedding 模型，CPU 完全够用。**默认装 CUDA 版 = 拉 5 GB 垃圾进镜像。**

**面试时能讲**：
> "我把 torch 从默认的 CUDA 版换成 CPU 版，下载量从 5 GB 砍到 600 MB，镜像从 3.5 GB 瘦到 2.4 GB。因为我们项目只用 bge-small 做 embedding，24 MB 的小模型在 CPU 上毫秒级，根本不需要 GPU。改法是在 requirements.txt 顶部加一行 `--extra-index-url https://download.pytorch.org/whl/cpu`。"

---

## 故事 25：Windows wheel 不能给 Linux 容器用（2026-10-08）

### 问题
为了避开容器内 pip 下载超时，我想先在 Windows 宿主机下载所有 wheel 到 `wheelhouse/`，再让容器离线安装。

宿主机跑：
```cmd
pip download -r requirements.txt -d wheelhouse -i https://mirrors.aliyun.com/pypi/simple/
```

下完了，`dir wheelhouse` 显示 100+ 个 whl。看文件名：
```
torch-2.14.1+cpu-cp314-cp314-win_amd64.whl
faiss_cpu-1.15.1-cp314-cp314-win_amd64.whl
pywin32-312-cp314-cp314-win_amd64.whl
...
```

### 分析
**三个维度都不匹配**：

| 维度 | Windows 宿主机 | Linux 容器 |
|---|---|---|
| 操作系统 | `win_amd64` | `manylinux*` |
| Python 版本 | `cp314`（本机 3.14） | `cp312`（容器 3.12） |
| ABI | Windows ABI | Linux ABI |

wheel 文件名后缀解读：
```
torch-2.14.1+cpu-cp314-cp314-win_amd64.whl
                ^^^^^  ^^^^^  ^^^^^^^^
                CPython 3.14  Windows amd64
```

**Docker 容器里**：
- 平台是 Linux（`manylinux`）
- Python 3.12（`cp312`）
- **一个都装不上**

### 方案
**放弃 wheelhouse 方案，回到容器内直接下载**（配合故事 23 的阿里云源 + 故事 24 的 CPU 版 torch，下载量降到 600 MB，能过）。

**如果非要本地下 wheel 给容器用**，必须用 `--platform` 明确指定：
```cmd
pip download -r requirements.txt -d wheelhouse ^
    --platform manylinux2014_x86_64 ^
    --platform manylinux_2_17_x86_64 ^
    --platform manylinux_2_28_x86_64 ^
    --platform any ^
    --python-version 3.12 ^
    --only-binary=:all: ^
    --timeout 300 --retries 10
```

### 为什么容易踩
- 开发时"下个 wheel 而已"看起来很简单
- 但 pip 默认下载**当前平台**的 wheel
- 宿主机是 Windows/Python 3.14，容器是 Linux/Python 3.12
- **两个环境的 wheel 不通用**

### 收获
**别在宿主机给容器下 wheel**——这是**跨平台工具链陷阱**。除非明确用 `--platform` + `--python-version` 指定目标平台，否则宿主机下下来的都是"自己用得上、容器用不了"的 whl。

**面试时能讲**：
> "为了避开容器内 pip 超时，我试过在 Windows 宿主机下载 wheel 让容器离线安装。结果发现宿主机 Python 3.14 下载的是 `win_amd64` + `cp314` 的 wheel，而容器是 Linux + Python 3.12，需要 `manylinux` + `cp312`，一个都装不上。后来我明白了：wheel 是平台相关的，跨平台必须用 `--platform` 明确指定。"

---

## 故事 26：MCP Server 硬编码 HF 缓存路径（2026-10-08）

### 问题
Docker 容器启动后，`/chat` 请求立刻报错：

```text
FileNotFoundError: [Errno 2] No such file or directory: 
'/root/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots'
```

MCP Server 启动失败 → 子进程崩溃 → `Connection closed` → FastAPI startup failed。

### 分析
看 `mcp_rag_server.py` 第 24 行：

```python
model_path = str(next(
    (Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots")
    .iterdir()
))
```

**代码硬编码了 HF 缓存路径**，直接去 `snapshots/` 目录里找模型文件。这要求：
1. 该目录必须存在
2. 里面必须有 hash 命名的子目录
3. 子目录里有 `config.json`、`model.safetensors` 等文件

**为什么容器里没有这个目录？**
- Dockerfile 里我删了构建期下载模型那行 `RUN python -c "...SentenceTransformer..."`（避网络超时）
- 改成运行时 volume 挂载：`./hf_cache:/root/.cache/huggingface`
- **但宿主机 `hf_cache/` 是空的** —— 从没跑过下载命令

### 方案（第一版：临时修）

**宿主机先把模型下到 `hf_cache/`**：

```cmd
cd /d D:\Projects\Agent
set HF_HOME=D:\Projects\Agent\hf_cache
set HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5')"
```

下载 192 MB（包含 13 个文件）。**因为 volume 挂载，容器内 `/root/.cache/huggingface/` 就能看到这些文件。**

### 方案（最终版：三级探测，2026-10-08）

**问题升级**：本地修好后，容器里又崩了。原因：
- MCP Server 是 `api_agent.py` 用 stdio 拉起的**子进程**
- `MultiServerMCPClient` 起子进程时，环境变量**可能不完整传递**
- 容器里 `HF_HOME` 为空时，代码 fallback 到项目内 `hf_cache/`（容器里不存在）

**改成三级探测**：

```python
import os
import sys
from pathlib import Path

# ===== HF 环境初始化（必须在 sentence_transformers 之前）=====
def _resolve_hf_home() -> str:
    if os.environ.get("HF_HOME"):           # 1. 显式环境变量
        return os.environ["HF_HOME"]
    container_default = Path("/root/.cache/huggingface")
    if container_default.exists():          # 2. 容器默认挂载点
        return str(container_default)
    return str(Path(__file__).resolve().parent.parent / "hf_cache")   # 3. 本地开发

_HF_HOME = _resolve_hf_home()
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ... import 之后

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name):
        # 让 HF 自己根据 HF_HOME 解析缓存
        self.model = SentenceTransformer(model_name)

embedding = LocalEmbeddings("BAAI/bge-small-zh-v1.5")
```

**同时 compose 里加环境变量双保险**：

```yaml
environment:
  - HF_ENDPOINT=https://hf-mirror.com
  - HF_HOME=/root/.cache/huggingface
  - HF_HUB_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
```

### 收获
**不要在代码里硬编码缓存路径。** 让库自己处理缓存逻辑（HF 的 `snapshot_download` 或 `SentenceTransformer` 都会自动管理）。硬编码路径的问题：
- 换环境就失效
- 依赖精确的目录结构
- 出错了不容易发现

**第二个教训**：**Dockerfile 里删了某行 RUN，要检查这行原本"副作用"（下载模型）有没有别的补偿机制。** 我删了 `RUN python -c "...下载模型..."`，改成 volume 挂载，但**忘了让用户先下载**。

**第三个教训**：**MCP 子进程的环境变量可能不完整。** 这是 fastmcp + langchain-mcp-adapters 底层 stdio 传输的一个特点。用**多级探测**（环境变量 > 容器默认路径 > 项目内路径）绕过。

**面试时能讲**：
> "我 Docker 化时把模型下载从构建期移到运行时 volume 挂载。但改完启动容器报 `FileNotFoundError: snapshots 目录不存在`。根因是 MCP Server 硬编码了 HF 缓存路径，而且子进程的环境变量可能不完整。我改成三级探测：环境变量 > 容器默认挂载点 `/root/.cache/huggingface` > 项目内 `hf_cache/`。这样本地和容器都能跑。"

---

## 故事 27：checkpoints.db 无 volume，容器重建丢记忆（2026-10-08）

### 问题
README 宣传"服务重启不丢历史"，但**只在进程重启场景下成立**。`docker compose down` 后：
- 容器销毁
- `rag_api/checkpoints.db` 随之消失
- 用户的历史对话全丢

**这是一个很容易忽略的边界**：restart ≠ down。

### 分析
原 `docker-compose.yml` 只挂了一个 volume：
```yaml
volumes:
  - ./hf_cache:/root/.cache/huggingface   # 只挂了 HF 缓存
  # 缺 checkpoints volume
```

`api_agent.py` 里的路径：
```python
checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
```
**落在源码目录里，宿主机看不到。**

### 方案

**① 宿主机建目录：**
```cmd
mkdir D:\Projects\Agent\data
```

**② 改 `api_agent.py`：**
```python
# checkpointer 持久化路径：优先用 DATA_DIR（容器里由 compose 设定），
# 否则用 rag_api/data/（本地开发）
data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
data_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = str(data_dir / "checkpoints.db")
async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
    ...
```

**③ 改 `docker-compose.yml`：**
```yaml
environment:
  - DATA_DIR=/app/rag_api/data
volumes:
  - ./hf_cache:/root/.cache/huggingface
  - ./data:/app/rag_api/data       # 新增
```

**④ 把旧的 db 挪到新位置：**
```cmd
move rag_api\checkpoints.db data\checkpoints.db
```

**⑤ `.gitignore` 加 `data/`：**
```
data/
```

### 验证
```cmd
:: 1. 问一个问题
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"MMR 是什么？\", \"thread_id\": \"p0b_test\"}"

:: 2. 完全销毁容器
docker compose down

:: 3. 重建
docker compose up -d

:: 4. 用代词问
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"它和普通相似度有什么区别？\", \"thread_id\": \"p0b_test\"}"
```

**第 4 步能理解"它"= MMR → 记忆持久化生效。**

**宿主机 `data/checkpoints.db` 大小会实时变化，证明写入生效。**

### 为什么挂目录，不挂文件
**不要**用这种写法：
```yaml
- ./checkpoints.db:/app/rag_api/checkpoints.db      # ❌ 危险
```

**原因**：如果宿主机 `./checkpoints.db` 不存在，Docker **会创建一个同名目录**挂进去，导致 SQLite 打不开。挂目录比挂文件稳。

### 收获
**Docker 持久化有三个层次，缺一不可**：
1. **代码层**：用环境变量（`DATA_DIR`）决定路径，不用硬编码
2. **compose 层**：volume 挂载到宿主机
3. **git 层**：`.gitignore` 忽略运行时文件（`data/`）

**"服务重启不丢"这句话要拆开看**：
- `docker compose restart` → 只重启进程，容器不销毁，DB 还在容器里 ✅
- `docker compose down` → 销毁容器，DB 一起消失 ❌
- `docker compose down && up` → 除非有 volume，否则记忆全丢

**面试时能讲**：
> "Docker 化的一个坑是 checkpoints 持久化。`AsyncSqliteSaver` 默认把 DB 写在容器内部，`docker compose down` 就全丢了。我的做法是用 `DATA_DIR` 环境变量决定路径，compose 里把 `./data` 挂到 `/app/rag_api/data`。本地开发 fallback 到 `rag_api/data/`。验证过：`docker compose down` 后 `up -d`，用代词问，Agent 还能回忆起上一轮。"

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
| 构建优化 | Dockerfile 指令顺序影响缓存效率；多阶段构建瘦身 |
| 协议理解 | stdio 通信要用 stderr 打日志 |
| 检索算法 | 混合检索要调权重，MMR 是 chunk 级多样性 |
| 生产级意识 | Agent 要有护栏，记忆要持久化 |
| 功能演进回归 | 加新功能要检查旧设计的假设是否还成立 |
| 幻觉治理 | Prompt 软约束 + 检索层硬降级，两层防护 |
| 评估方法 | 单一指标会误导，要多个指标一起看 |
| 记忆持久化 | 让 checkpointer 管历史，不要在框架外面手动糊一层 |
| 架构重构 | 发现职责重叠就统一到单一真相源 |
| 安全防护 | 服务端日志记详细，客户端响应给友好提示 |
| 缓存正确性 | 缓存不是免费午餐，命中率结构性为 0 就删掉 |
| 依赖管理 | requirements.txt 必须经过干净环境验证 |
| **磁盘与 Docker 环境** | **Docker 构建失败先查磁盘空间；WSL2 vhdx 默认在 C 盘要迁走** |
| **依赖下载与平台兼容** | **CPU 版 torch 能砍 90% 下载量；容器内下载比宿主机稳** |
| **跨平台 wheel 陷阱** | **wheel 是平台相关的，宿主机给容器下 wheel 必须指定 `--platform`** |
| **模型缓存与容器** | **别硬编码 HF 缓存路径；三级探测让本地和容器都能跑** |
| **checkpoints 持久化** | **`docker compose down` 会丢记忆，必须挂 volume 到宿主机** |

**每个问题的解决过程都体现了同一个原则：先分析根本原因，再选择方案，最后量化效果。**