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

**每个问题的解决过程都体现了同一个原则：先分析根本原因，再选择方案，最后量化效果。**