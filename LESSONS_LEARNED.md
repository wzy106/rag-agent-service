# 项目踩坑与复盘

> 记录项目开发过程中遇到的问题、分析过程和解决方案。
> 每个故事按"问题 → 分析 → 方案 → 收获"结构。
> 最后更新：2026-10-16（新增「迭代故事（面试主线）」：把 10 条迭代弧提炼成面试主线；订正性能数字；修复 `/history` 越权 + 多租户隔离；新增故事 35：测试与 CI）

## 快速索引

| 类别 | 对应故事 |
|---|---|
| 模型 API 限制 | 故事 1 |
| 依赖冲突 | 故事 2 |
| 网络问题 | 故事 3 |
| 异步编程 | 故事 4、5 |
| 用户体验 | 故事 6 |
| 构建优化 | 故事 7、26、33 |
| 协议理解 | 故事 8 |
| 检索算法 | 故事 9、10 |
| 生产级意识 | 故事 11 |
| 功能演进回归 | 故事 12 |
| 幻觉治理 | 故事 14 |
| 评估方法 | 故事 15 |
| 记忆持久化 | 故事 13、16 |
| 架构重构 | 故事 17 |
| 安全防护 | 故事 18、32 |
| 缓存正确性 | 故事 19 |
| 依赖管理 | 故事 20 |
| **磁盘与 Docker 环境** | **故事 21、22** |
| **依赖下载与平台兼容** | **故事 23、24** |
| **跨平台 wheel 陷阱** | **故事 25** |
| **模型缓存与容器** | **故事 26** |
| **checkpoints 持久化** | **故事 27** |
| **评估代码分叉** | **故事 28** |
| **阈值拍脑袋定 / 重标定** | **故事 29** |
| **Rerank 收益被语料压制** | **故事 30** |
| **测试集太小判别力不足** | **故事 31** |
| **接口裸奔** | **故事 32** |
| **改代码每次 rebuild** | **故事 33** |
| **限流未落地** | **故事 34** |
| **测试与 CI** | **故事 35** |

---

## 迭代故事（面试主线）

> 项目不是一次做成的，是**一轮轮改出来的**。下面 10 条按"**起点 → 怎么发现的 → 迭代成了什么 → 认知**"提炼。
> 面试时用它串主线；细节去查对应的编号故事。
>
> **为什么单拎出来**：细节故事是按"坑"编号的（东一个西一个），但面试官想听的是**"你是怎么一步步变好的"**——那是另一条线。

### 1. 阈值：从"拍脑袋"到"标定"
- **起点**：`MAX_DISTANCE = 1.1`，随手定的，说不出依据
- **怎么发现的**：发现幻觉率 9.1%，但**没人能解释这个数哪来的**
- **迭代**：正负样本距离分布 → 扫 101 个阈值画 PR 曲线 → 取 F1 最大点 → 1.0；语料扩大后**重标定** → 0.96
- **认知**：阈值不该"调"出来，该"标定"出来。**有实验支撑的数字才敢在面试里报**（→ 故事 29）

### 2. 测试集：从"全是 100%"到"有判别力"
- **起点**：2 篇文档 / 30 条测试项——所有策略召回率都接近 100%，看起来完美
- **怎么发现的**：算了下**随机基线 = 84.1%**——随便抽 3 个 chunk 就能命中 84%
- **迭代**：语料扩到 98 篇、测试集扩到 249 条 → 随机基线降到 3.0%
- **认知**：**没有随机基线的指标是自欺**。天花板效应会淹没所有策略差异——Rerank 的增益从"+0.04 精确率"（被淹没）变成"+4.4% 召回率"（真实）（→ 故事 31、32）

### 3. 检索：从"单一"到"两阶段"
- **起点**：纯向量相似度检索
- **怎么发现的**：口语化提问和文档术语之间有语义鸿沟；MMR 在精确匹配任务上只有 37.8% 召回
- **迭代**：加 BM25 → **等权 RRF 反而比纯向量差** → 加权 0.7/0.3 → 再加 CrossEncoder 精排成两阶段
- **认知**：**等权融合是陷阱**（较弱的一路被拉高了话语权）；双塔快但粗、交叉编码器精但慢 → 组合兼顾（→ 故事 9、10、30）

### 4. 记忆：从"双源打架"到"单一真相源"
- **起点**：`InMemorySaver`（重启就丢）+ **手动从 MySQL 读历史拼进 messages**
- **怎么发现的**：两套记忆和 checkpointer 的 state **打架**——消息格式混用、历史重复累积
- **迭代**：统一到 LangGraph checkpointer（`AsyncSqliteSaver` 持久化）；MySQL **降级为展示层**
- **认知**：**记忆只能有一个真相源**；一致性 bug 比性能 bug 更难查（→ 故事 16、17）

### 5. 缓存：从"加上"到"删掉"
- **起点**：加了 Redis 缓存，key 里带 `context_version`
- **怎么发现的**：实测 `context_version` **每轮 +2**，命中率**结构性为 0**
- **迭代**：删掉
- **认知**：**加缓存前先算命中率**。"优化"可能是负优化——**删代码也是迭代**（→ 故事 19、20）

### 6. 性能：从"每次重开子进程"到"常驻 session"
- **起点**：用 `client.get_tools()`；文档还写着"MCP Server 常驻"——**一条没验证过的假设**
- **怎么发现的**：计时发现**每次工具调用 35~44 秒**；读库源码发现它"每次调用都新建 session"（等于重开子进程 + 重载全部模型）
- **迭代**：改 `client.session()` + `load_mcp_tools` 常驻；工具改 async + `anyio.to_thread`；加读超时 + 启动预热
- **认知**：**不要相信没测过的数字**——文档里"单次 2~3 秒""Rerank 50~100ms"全是假的，实测差 1~2 个数量级（→ 故事 26、35）

### 7. 安全：从"裸奔"到"授权"
- **起点**：接口完全无鉴权
- **迭代 1**：加 API Key——但**只做了认证**（只判断"key 对不对"，不知道"是谁"）
- **怎么发现的**：`/history` 无鉴权 + `thread_id` 有默认值 → **任何人按 thread_id 就能读别人的会话**（IDOR / 水平越权）
- **迭代 2**：`/history` 加鉴权 + `thread_id` 必填 + 多 key 租户隔离 + 恒时比较
- **认知**：**认证 ≠ 授权**。给接口挂个 `Depends` 是"假修"——根因是**系统里没有"身份"概念**（→ 故事 32）

### 8. 工程化：从"零测试"到"有 CI"
- **起点**：所有验证都是手工的（起服务 + curl + 肉眼）
- **怎么发现的**：写测试时发现 `eval/`、`scripts/` **import 时就执行**——根本没法被 pytest 导入
- **迭代**：分单元/集成两层 + 打桩（假 Agent/MySQL/Redis）+ GitHub Actions
- **认知**：**可测试性是设计出来的**；CI 是"作品"和"工程"的分界线（→ 故事 35）

### 9. 部署：从 3.51GB 到 2.42GB
- **起点**：默认 CUDA 版 torch + 在构建期下载模型
- **怎么发现的**：构建要 33 分钟、镜像 3.5GB，而且 C 盘被撑爆了
- **迭代**：多阶段构建 + CPU 版 torch（下载量 5GB → 600MB）+ 模型挂 volume 运行时下载
- **认知**：**构建期不该做运行时的事**（→ 故事 21、22、33）

### 10. 代码组织：从"两套代码"到"共用一份"
- **起点**：评估脚本和服务端**各写一份**检索逻辑
- **怎么发现的**：意识到评估和服务端可能**分叉**——评估说好、线上却差
- **迭代**：抽 `rag_core/search.py`，服务端和评估脚本共用同一份
- **认知**：**评估跑的就是部署的代码**，否则指标没有意义（→ 故事 28）

### 面试主线话术（把这些串成一段）

> "这个项目不是一次做成的——**我大概迭代了十来轮**。最典型的三条线：
> ① **评估线**：我一开始只有 2 篇文档、30 条测试项，所有策略看起来都是 100%，直到我算了随机基线发现是 **84%**——才意识到是天花板效应。扩语料到 98 篇、测试集到 249 条后，基线降到 3%，Rerank 的真实增益才显现出来。
> ② **安全线**：我先加了 API Key 鉴权，但后来意识到那**只做了认证、没做授权**——`/history` 没鉴权加上 `thread_id` 有默认值，等于任何人能读别人的会话。修的时候才明白，光加鉴权是假修，根因是系统里没有'身份'概念。
> ③ **性能线**：文档写着'单次请求 2~3 秒'，但我从来没测过。真去计时发现**每次工具调用要 35~44 秒**——根因是 MCP 每次调用都重开子进程。改完降到 7 秒。
> **这三条线的共同点**：都不是'我加了个功能'，而是**'我发现原来那样是错的，然后改对了'**。"

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
    async with client.session("rag") as session:      # 常驻 session
        tools = await load_mcp_tools(session)
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

> **后续订正（2026-10-16）**：上面的示例用了 `client.get_tools()`——后来发现它**每次工具调用都会新建 session**，等于每次调用都重新拉起 MCP 子进程并重载全部模型，实测**每次工具调用 35~44 秒**。已改为**常驻 session**（`async with client.session("rag") as session:` + `load_mcp_tools(session)`），子进程整个服务生命周期只 spawn 一次。

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

### 更新（2026-10-13）
代码 COPY 进镜像导致"改一行就 rebuild"，磁盘也越吃越多。**最终方案**：开发期 volume 挂载代码/语料/索引，改代码 `docker compose restart` 即可。详见故事 33。

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

### 更新（2026-10-12）
在 249 条测试集上，混合检索召回率 **52.6%** > 纯相似度 50.6%（**+2.0%**）。说明加权 RRF 的价值在**更大的测试集**上被验证了。

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

### 更新（2026-10-12）
在 249 条测试集上，MMR 召回率只有 **37.8%**（低于相似度 50.6%）——**在"精确匹配"任务上吃亏**。但多样性 2.79（高于相似度 2.63）。**这就是"多样性导向"的代价**。

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
MAX_DISTANCE = 0.96

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

### 更新（2026-10-10）
上面"选 1.1"是拍脑袋定的，只有几条样例，没有统计支撑。**当时方案**：通过正负样本距离分布 + PR 曲线标定为 1.0。详见故事 29。

### 更新（2026-10-12）
语料从 550 chunk 扩到 583 chunk 后，已重新标定 `MAX_DISTANCE = 0.96`。新标定结果：P=0.990、R=1.000、F1=0.995；正样本 100 条（test_set_v2.json，seed=42）。详见故事 29。

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

### 实测结果（**249 条测试集，98 篇文档 / 583 chunk**）

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 普通相似度 | 50.6% | 0.23 | 0.39 | 2.63 |
| MMR | 37.8% | 0.15 | 0.33 | 2.79 |
| MMR + 去重 | 40.6% | 0.14 | 0.34 | **3.00** |
| 混合检索 | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |
| 随机基线 | **3.0%** | — | — | — |

### 关键发现
- **随机基线从 84.1% 降到 3.0%**：语料从 2 篇扩到 98 篇后，判别空间从 15.9 个点扩到 54.0 个点
- **精确率和多样性呈 trade-off**：想要多角度信息，就要接受部分结果不直接相关
- **MMR 在精确匹配上吃亏**：召回率只有 37.8%，但多样性最高
- **选策略看业务**：追求精确用混合+Rerank，需要多角度用 MMR+去重

### 收获
**单一指标会误导。** 只看召回率以为三种策略一样好，加上精确率和多样性才看出它们的 trade-off。**评估 RAG 要多个指标一起看。**

### 更新（2026-10-12）
上面的数字已更新为 **249 条测试集 + 98 篇文档 / 583 chunk** 的新版本。**判别空间从 15.9 个点扩到 54.0 个点**，各策略的差异才真正显现。详见故事 31。

### 已知不足
- **文档级召回而非 chunk 级**：用 `source`（相对路径）判断命中，同一文档的多个 chunk 算命中。粒度还可以更细
- **source 级评估可能高估效果**：估计高估 8~14 个百分点；后续需要 chunk 级标注或人工复核
- **chunk 级标注成本高**：需要人工标注每个问题的"正确 chunk id"，耗时大
- **没有端到端指标**：检索对了不等于回答对了。应该加 LLM-as-judge 或人工标注的 answer correctness

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
- CI 流水线同理（**2026-10-16 补充**：CI 已落地，`pip install -r requirements-dev.txt` 会把这类缺失暴露在 push 阶段）

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

（`redis` 现在正式用于限流，不再是"预留"）

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
  - DATA_DIR=/app/data
volumes:
  - ./hf_cache:/root/.cache/huggingface
  - ./data:/app/data       # 新增
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
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -H "Authorization: Bearer your-secret-key-here" -d "{\"message\": \"MMR 是什么？\", \"thread_id\": \"p0b_test\"}"

:: 2. 完全销毁容器
docker compose down

:: 3. 重建
docker compose up -d

:: 4. 用代词问
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -H "Authorization: Bearer your-secret-key-here" -d "{\"message\": \"它和普通相似度有什么区别？\", \"thread_id\": \"p0b_test\"}"
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
> "Docker 化的一个坑是 checkpoints 持久化。`AsyncSqliteSaver` 默认把 DB 写在容器内部，`docker compose down` 就全丢了。我的做法是用 `DATA_DIR` 环境变量决定路径，compose 里把 `./data` 挂到 `/app/data`。本地开发 fallback 到 `rag_api/data/`。验证过：`docker compose down` 后 `up -d`，用代词问，Agent 还能回忆起上一轮。"

---

## 故事 28：评估代码和部署代码分叉（2026-10-09）

### 问题
项目里有两份 `hybrid_search`：
- `mcp_test/mcp_rag_server.py`：有 `max_distance` 降级
- `eval/hybrid_search_test.py`：没有降级

**所以 README 里"混合检索加权 RRF"的数字，描述的不是部署的系统。**

更糟的是：评估脚本没有随机基线。"三种策略都是 100% 召回率"看起来很漂亮，但没有对照系——不知道这个 100% 含金量是多少。

### 分析
**为什么会出现分叉**：开发时"顺手复制一份改改"最省事，不用考虑模块化。这在单个文件里没问题，但一旦有**多个消费方**（服务端 + 评估脚本 + 未来的测试），分叉就变成隐藏 bug。

**为什么必须修**：
- 面试官问"你 README 里的数字怎么来的"，答不出可复现的脚本就是扣分
- 招聘方最关心的不是"结果多好"，而是"结果可信吗"
- 无随机基线 = 不知道自己离天花板有多远

### 方案

**① 抽公共模块 `rag_core/search.py`**：
```python
# rag_core/search.py
def hybrid_search(query, vectorstore, bm25, all_docs, k=3, ...):
    ...
```
服务端和评估都 `from rag_core.search import hybrid_search`。

**② Dockerfile 加 COPY**：
```dockerfile
COPY rag_core/ ./rag_core/
```

**③ `mcp_rag_server.py` 加 sys.path hack**（脚本在 `mcp_test/` 子目录，找不到项目根的 `rag_core`）：
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import hybrid_search, ...
```

**④ 评估脚本加随机基线**（组合公式精确计算，不用模拟）：
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

**⑤ 删掉旧的 `eval/hybrid_search_test.py`。**

### 效果（30 条测试集时的数字）

跑 `python eval/eval_retrieval.py`：
```
[随机基线] 召回率：84.1%

策略                              召回率      精确率      MRR      多样性
普通相似度                        100.0%     0.87     0.96     1.37
MMR                              100.0%     0.83     0.97     1.43
MMR + 按来源去重                  100.0%     0.77     0.97     1.57
混合检索（向量+BM25+RRF）          96.7%     0.86     0.94     1.30
```

### 更新（2026-10-12）
**后来扩语料 + 扩测试集**，随机基线降到 **3.0%**，判别空间从 15.9 个点扩到 **54.0 个点**。详见故事 31。

### 收获
**任何被多个消费方引用的逻辑，都要抽公共模块。** 复制粘贴在当时省事，但埋下"两份实现漂移"的雷。

**评估脚本必须能自己跑出所有报出的数字。** 别人 clone 项目后跑一条命令，能验证 README 里每张表的每个数字——这是"可信"的最基本要求。

**随机基线是评估的地板，不是天花板。** 没有地板就无法判断"100% 召回"到底含金量多高。

**面试时能讲**：
> "我评估脚本和服务端各写了一份 hybrid_search，参数还不一样——评估那份缺距离降级。发现后把检索逻辑抽到 rag_core/search.py，两边都 import，Dockerfile 也加了 COPY。同时给评估脚本加了随机基线，用组合公式精确计算。"

---

## 故事 29：`MAX_DISTANCE` 阈值拍脑袋定（2026-10-10，更新于 2026-10-12）

### 问题
降级阈值 `MAX_DISTANCE = 1.1` 是早期凭感觉定的——跑了几条样例，正常问题距离 0.58~0.89，无关问题 1.15~1.34，中间取个约数。**没有任何统计支撑，也没有评估覆盖。**

面试官问"为什么是 1.1"，只能说"实测差不多在这"，答不出"为什么不是 1.0 或 1.2"。

### 分析
**"拍脑袋定超参数"是常见的工程债**：
- 单点验证（跑几条样例）看起来够用，但样本量太小，受偶发因素影响大
- 没有正负样本对照，不知道"距离分布是否真的能分开两类"
- 没有 PR 曲线，不知道阈值在哪个点上是权衡最优
- 阈值变了没有回归测试，改了不知道好坏

**为什么必须标定**：
- 阈值直接决定"系统什么时候拒答"——这是 RAG 幻觉防护的最后一关
- 面试时"我做了标定实验"和"我拍脑袋定的"是两个量级的回答
- 标定脚本本身也是"可复现性"的一部分

### 方案

**① 构造正负样本**：
- 正样本 100 条：从 `test_set_v2.json` 抽样，seed=42
- 负样本 25 条：手动造"知识库明显没有"的问题（天气、美食、电影、体育等）
- 存成 `eval/negatives.json`

**② 对每条问题算 `best_distance`**：
```python
def best_distance(query):
    results = vectorstore.similarity_search_with_score(query, k=1)
    return results[0][1] if results else float("inf")
```

**③ 画距离分布直方图**：
- 正样本蓝色，负样本红色，叠加看重叠
- 结果：正样本 0.363~0.958（中位数 0.709），负样本 0.813~1.420（中位数 1.176）
- **重叠区 [0.813, 0.958]**——判别力天花板，任何阈值都必然有误判

**④ 扫 101 个阈值（0.5~1.5）算 P/R/F1**：
```python
def evaluate_threshold(thr):
    tp = sum(1 for d in pos_dists if d <= thr)
    fp = sum(1 for d in neg_dists if d <= thr)
    fn = len(pos_dists) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
```

**⑤ 选阈值**：

| 阈值 | P | R | F1 | 说明 |
|---|---|---|---|---|
| 1.10（原值） | 0.909 | 1.000 | 0.952 | 幻觉率 9.1% |
| 1.00（旧标定） | 0.967 | 0.967 | 0.967 | 幻觉率 3.3%，漏答 3.3% |
| **0.96（当前）** | **0.990** | **1.000** | **0.995** | **幻觉率约 1.0%，漏答 0%** |

**为什么选 0.96 而不是旧 1.00**：
- 正样本距离上界是 0.958，取 0.96 可以全收正样本
- 比 1.0 更紧，能减少负样本误入上下文
- RAG 场景用户对"编造"的容忍度远低于"不知道"
- 新标定 P=0.990、R=1.000、F1=0.995，整体优于旧阈值

### 效果
- **幻觉率**（负样本被误答）：9.1% → **约 1.0%**
- **漏答率**（正样本被拒答）：0% → **0%**
- 精确率 99.0%，召回率 100%，F1 0.995

### 产出
- `eval/calibrate_distance.py`（标定脚本，可复现）
- `eval/negatives.json`（25 条负样本）
- `eval/output/distance_distribution.png`（距离分布图）
- `eval/output/pr_curve.png`（PR 曲线）

### 收获
**超参数要么标定，要么写清楚是拍的。** 拍脑袋本身不丢人，但**在面试里说"我是标定的"和"我猜的"，是两种信誉**。

**标定脚本比阈值本身更有价值。** 阈值可能随着语料变化而调，但**标定的方法论**可以一直用。别人 clone 项目后跑 `python eval/calibrate_distance.py`，能验证你的每一个数字。

**RAG 场景的业务权衡要显式表达。** 不是"最优 F1"，而是"能接受的幻觉率"——这是产品级思维，不是纯技术。

**面试时能讲**：
> "MAX_DISTANCE=0.96 不是拍脑袋定的。我构造了 100 条正样本和 25 条负样本，对每条跑 best_distance，画分布直方图看分离点，再扫阈值从 0.5 到 1.5 画 PR 曲线。最终取 0.96——因为这个点上精确率 99.0%，召回率 100%，F1 0.995；正样本距离上界是 0.958，0.96 刚好全收正样本，同时比旧阈值 1.0 更紧，能减少负样本误入。标定脚本和结果图都在仓库里，可以复现。"

---

## 故事 30：Rerank 收益被语料规模压制（2026-10-11，更新于 2026-10-12）

### 问题
为了提升检索精度，我加了两阶段检索：混合检索召回 Top-20 → Reranker（bge-reranker-base）精排 Top-3。

**预期**：精确率 0.86 → 0.92+，MRR 0.94 → 0.99+。

**实际（30 条测试集时）**：精确率 0.86 → **0.90**，MRR 0.94 → **0.97**。

**提升有效，但远没到预期。**

### 分析
**Rerank 本身没问题**——它确实把更相关的文档排到了前面（精确率 +0.04，MRR +0.03）。

**问题在语料规模**：
- 只有 2 篇文档、102 个 chunk
- 每条 query 的期望 source 只有 1~2 个（甚至有些 query 期望命中两篇）
- 随机基线召回率 84.1%——意味着"瞎抽 3 个"就已经命中 84%
- **真实判别空间只有 15.9 个点**

**当语料小时，天花板效应很明显**：
- 混合检索已经能拿到大部分正样本（96.7% 召回）
- Reranker 能优化的空间就那么大——它只能在"已经召回的 20 条里"重排
- 如果这 20 条里真正相关的只有 1~2 条，Reranker 再准也只能把其中 1 条排到 Top-1，另外 2 个位置还是会被不太相关的占据

**类比**：考试范围只有 10 页书，满分 100 分。你从 80 分提到 90 分，已经接近天花板了——不可能再提到 98。

### 方案（短期）
- 加了 Reranker，精确率 0.86 → 0.90
- 用 `recall_k=20, final_k=3` 的两阶段 pipeline
- 懒加载设计（第一次调用才加载 约 1.1 GB 模型，不拖慢服务启动）
- **性能代价（2026-10-16 实测）**：CPU 上精排 20 候选约 **2~7 秒/次**（负载高时可达 13 秒），是整条检索链路里最慢的一环。这里早先写过"单次 50~100 ms"，但**从未实测**，差了 1~2 个数量级——面试如果被追问"怎么测的"会露馅。

### 方案（长期，已完成）
**扩语料到 98 篇 / 583 chunk**，测试集扩到 249 条。新结果：

| 策略 | 召回率 | 精确率 | MRR | 多样性 |
|---|---|---|---|---|
| 混合检索 | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |

**Rerank 增益从 +0.04 精确率 → +4.4% 召回率、+0.05 MRR。**

**为什么扩语料后增益变大**：
- 判别空间从 15.9 个点扩到 54.0 个点
- 混合检索的 Top-20 里"相关但排名靠后"的 chunk 更多了
- Reranker 有更多空间把关键 chunk 顶上来

### 收获
**Rerank 是完整 RAG pipeline 的最后一块拼图。** 召回 → 融合 → 精排，缺了精排就像"做菜没上调料"。

**但精排的收益被语料规模约束**：
- 语料小 → 天花板低 → Rerank 可优化空间小
- 语料大 → 天花板高 → Rerank 才能真正发挥作用
- **再好的精排也救不了小语料**

**面试时要主动说这个"从小语料到大语料"的迭代故事。** 这比"我加了 Rerank，指标提升 X%"更可信——**面试官会觉得你知道自己在说什么**。

**面试时能讲**：
> "我加了两阶段检索：混合检索召回 Top-20，用 bge-reranker-base 交叉编码器精排 Top-3。第一版语料只有 2 篇文档、测试集 30 条，精确率从 0.86 提到 0.90，看起来提升不大。我分析了一下——随机基线 84.1%，判别空间只有 15.9 个点，天花板效应太明显。**后来扩语料到 98 篇、测试集到 249 条**，随机基线降到 3.0%，判别空间扩到 54.0 个点，Rerank 的增益才真正显现——召回率从 52.6% 提到 57.0%，MRR 从 0.42 提到 0.47。这是个真实的迭代故事：**不是 Rerank 不好用，是之前的测试集不够大，看不出它的价值。**"

---

## 故事 31：测试集太小，判别力不足（2026-10-11，更新于 2026-10-12）

### 问题
从故事 15 到故事 30，"测试集太小"这个隐患一直在。

**具体表现**：
- 30 条测试项，只有 2 篇文档 / 102 chunk
- 随机基线召回率 **84.1%**——意味着"瞎抽 3 个"就已经命中 84%
- 判别空间只有 **15.9 个点**（100% - 84.1%）
- 各策略的召回率**全 100%**——看不出差异
- Rerank 只提升 +0.04 精确率——被天花板效应压制

**面试官会怎么问**：
- "你 30 条测试项，一条样本占 3.3%，统计上有什么意义？"
- "随机基线 84.1%，你的 100% 到底比瞎抽好多少？"
- "各策略都是 100%，说明测试集区分度不够吧？"

**这是项目最致命的短板。** 数字好看（全 100%）但**不可信**——因为测试集太简单。

### 分析
**"测试集太小"有 3 个层次的问题**：

**① 语料太小 → 天花板低**
- 2 篇文档，随机抽 3 个 chunk 命中期望 source 的概率是 84.1%
- 判别空间只有 15.9 个点，任何策略都容易摸到天花板

**② 测试项太少 → 统计无意义**
- 30 条测试项，3% 的差异 = 1 条样本的波动
- 97% vs 93% 这种"差异"完全是噪声

**③ 文档级粒度 → 判别力弱**
- 用 `source`（文件名）判断命中，只有 2 篇文档
- 命中太容易，含金量低

### 方案

**① 扩语料 2 → 98 篇**

用脚本把三个项目笔记（README、PROJECT_NOTES、LESSONS_LEARNED）+ LangChain 学习笔记，**按标题拆成多个独立文档**：

```python
# scripts/split_notes.py
def split_by_h2(content: str) -> list:
    """按 '## xxx' 二级标题拆分。返回 [(title, body), ...]"""
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    # ... 按标题边界切分
```

**结果**：
```
docs/
├── langchain/         (12 篇)
├── notes/             (15 篇，原 16 篇的 07 被拆了)
├── notes_interview/   (25 篇，从 07 拆出)
├── readme/            (13 篇)
└── stories/           (33 篇)
```

**98 篇文档 / 583 chunk。**

**关键设计**：**把一个大文件按 `##` 拆分**，而不是保持大文件：
- 拆之前：`notes/07` 一个文件 74 chunks，占 13.5%
- 拆之后：25 个独立文件，每个 3 chunks——**判别力大幅提升**

**② 改 `mcp_rag_server.py` 用 `rglob` 扫子目录**

```python
for md_file in docs_dir.rglob("*.md"):   # 递归扫子目录
    text = md_file.read_text(encoding="utf-8")
    rel_path = md_file.relative_to(docs_dir).as_posix()
    documents.append(Document(page_content=text, metadata={"source": rel_path}))
```

**关键**：`glob` → `rglob`，`source` 用相对路径（避免不同子目录同名文件冲突）。

**③ 测试集自动生成 30 → 249 条**

`scripts/generate_testset.py`：

```python
PROMPT_TEMPLATE = """你是一个 RAG 测试集生成助手。

下面是一篇技术文档的两个片段。请基于这些内容生成 3 个不同的中文问题，要求：
1. **问题不能直接复制片段中的原文措辞**
2. **问题要能被这些片段回答**
3. **3 个问题覆盖不同角度**
4. 每个问题 8~30 个字
5. 用口语化的表达，像真实用户会问的
"""

async def generate_questions(source: str, chunks: list, sem: asyncio.Semaphore):
    async with sem:
        # 挑两个最长 chunk
        sorted_chunks = sorted(chunks, key=lambda c: len(c.page_content), reverse=True)
        picked = sorted_chunks[:CHUNKS_PER_DOC]
        if len(picked) < 2:
            return None
        prompt = PROMPT_TEMPLATE.format(...)
        resp = await client.chat.completions.create(...)
        questions = json.loads(content)
        return {"source": source, "questions": questions[:3]}
```

**流程**：
- 遍历所有 chunk，按 source 分组
- 每篇挑 2 个最长 chunk
- 调 DeepSeek API 生成 3 个问题
- 标注 `expected_sources = 该文档的 source`

**成本**：约 ¥0.2（98 篇 × 1 次 API 调用）
**时间**：并发 5，约 1~2 分钟
**结果**：249 条测试项（13 篇极短文档被跳过）

**④ 评估脚本加随机基线**（组合公式）

```python
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

### 效果对比

**旧语料 / 旧测试集**：
```
[随机基线] 召回率：84.1%
策略                              召回率      精确率      MRR      多样性
普通相似度                        100.0%     0.87     0.96     1.37
MMR                              100.0%     0.83     0.97     1.43
MMR + 按来源去重                  100.0%     0.77     0.97     1.57
混合检索                          96.7%     0.86     0.94     1.30
混合检索 + Rerank                  96.7%     0.90     0.97     1.17
```

**新语料 / 新测试集**：
```
[随机基线] 召回率：3.0%
策略                              召回率      精确率      MRR      多样性
普通相似度                        50.6%     0.23     0.39     2.63
MMR                              37.8%     0.15     0.33     2.79
MMR + 按来源去重                  40.6%     0.14     0.34     3.00
混合检索                          52.6%     0.24     0.42     2.57
混合检索 + Rerank                  57.0%     0.25     0.47     2.56
```

**关键变化**：

| 指标 | 旧 | 新 | 提升 |
|---|---|---|---|
| 语料 | 2 篇 | **98 篇** | ×49 |
| Chunk | 102 | **583** | ×5.7 |
| 测试集 | 30 条 | **249 条** | ×8.3 |
| **随机基线** | 84.1% | **3.0%** | ↓81.1 个点 |
| **判别空间** | 15.9 个点 | **54.0 个点** | ×3.4 |
| Rerank 增益 | +0.04 精确率 | **+4.4% 召回、+0.05 MRR** | 更真实 |

### 收获
**测试集的大小决定了评估的可信度。** 30 条样本上的"100% 召回"和 249 条样本上的"57.0% 召回"，**后者才是真实的**——因为前者只是被天花板效应掩盖了。

**随机基线是评估的"地板"。** 没有地板就不知道"100%"有多少含金量。旧测试集 84.1% 的地板说明：语料太小，判别空间太窄。

**拆大文件能显著提升文档级召回的判别力。** `notes/07` 一个文件 74 chunks，所有 25 道面试题都算"同一篇文档"。拆成 25 个独立文件后，判别力大幅提升。

**测试集生成可以自动化。** 用 DeepSeek API 批量生成问题，成本约 ¥0.2，1~2 分钟出 249 条。**人工筛选**只是可选的质量把关，不是必需的。

**面试时要主动讲这个"从 30 条到 249 条"的迭代故事。** 这展示了：
- 你知道小样本评估的局限
- 你有能力用脚本批量扩测试集
- 你能对比新旧数字，而不是只报最好的

**面试时能讲**：
> "我第一版 RAG 评估只有 30 条测试项，语料只有 2 篇文档。数字看起来很漂亮——所有策略都是 100% 召回率。但我算了一下随机基线——84.1%，意味着随便抽 3 个 chunk 就能命中 84%。这说明我的测试集判别空间只有 15.9 个点，**所有策略都摸到了天花板，看不出差异**。后来我把三个项目笔记按标题拆成多个独立文档、再补上 LangChain 学习笔记，总语料 98 篇 583 chunk。然后写了个脚本用 DeepSeek API 批量生成问题——每篇挑 2 个最长 chunk 生成 3 个问题，成本约两毛钱。扩到 249 条测试项后，随机基线降到 3.0%，判别空间扩到 54.0 个点。**这时候数字才真实**：相似度 50.6%、混合检索 52.6%、加 Rerank 57.0%。Rerank 的真实增益也显现出来——召回率 +4.4%、MRR +0.05。"

---

## 故事 32：接口裸奔（安全防护 2，2026-10-13）

### 问题
`/chat` 和 `/chat/stream` 从一开始就是**完全裸奔**的——任何人只要知道 URL 就能调。项目做了记忆持久化、错误信息脱敏、幻觉防护，但最外层的大门一直开着。

面试官一问"你的接口怎么防滥用"，只能答"没做"。

### 分析
**为什么没做**：
- 本地跑着玩，没人会攻击
- 做鉴权要配环境变量、写校验逻辑、改接口签名，看起来"没必要"

**为什么必须做**：
- 面试官问的不是"有没有被攻击"，而是"**有没有生产级意识**"
- 玩具项目：接口直接暴露
- 生产项目：任何对外接口都要考虑"谁能调、调多少次、出错怎么处理"
- 加了不只是防攻击，更是**证明你知道生产环境长什么样**

### 方案

**① FastAPI 侧：`Header` + `Depends`**

```python
from fastapi import FastAPI, HTTPException, Header, Depends

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
```

**② 接口挂上依赖**

```python
@app.post("/chat")
async def chat(payload: ChatRequest, token: str = Depends(verify_key)):
    ...
```

请求进 `/chat` 前，FastAPI 先跑 `verify_key`。抛异常就拦截，正常返回才进业务。返回的 `token` 供后续限流使用。

**③ `docker-compose.yml` 注入**

```yaml
environment:
  - API_KEY=${API_KEY:-dev-key-change-me}
```

从宿主机 `.env` 读，读不到用默认值。

**④ `.env` 加一行**

```text
API_KEY=your-secret-key-here
```

### 验证

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

三条全过。

### 关键设计

| 点 | 说明 |
|---|---|
| `Header` | FastAPI 自动从请求头取值 |
| `Depends` | 依赖注入，一次定义、多处复用 |
| `401` vs `403` | 401 = 没带凭证；403 = 带了但不对 |
| `API_KEY` 环境变量 | 密码外置，代码里不出现真实密钥 |
| `${API_KEY:-默认}` | compose 从宿主机读，注入容器 |
| `/history` 保持公开 | 展示层不需要鉴权 |
| `verify_key` 返回 token | 供后续限流用 |

### 收获
**鉴权不是为了防真实攻击，是为了证明你有生产级意识。** 面试官看的是"你知道生产环境长什么样"，而不是"你的玩具项目有没有被入侵"。

**用 `Depends` 而不是手动调**：鉴权逻辑抽出来，`/chat` 和 `/chat/stream` 共用一份，改一处两端生效。这是"单一职责"在框架层的体现。

**面试时能讲**：
> "接口加了 API Key 鉴权。用 FastAPI 的 `Header` + `Depends` 做依赖注入，`/chat` 和 `/chat/stream` 共用同一个 `verify_key`，鉴权逻辑和业务逻辑解耦。`API_KEY` 从环境变量读——本地 `.env` 一份，生产服务器一份，代码里不出现真实密钥。`docker-compose.yml` 里用 `${API_KEY:-dev-key-change-me}` 语法从宿主机读取，读不到用默认值。"

---

## 故事 33：改代码每次 rebuild（2026-10-13）

### 问题
项目把 `api_agent.py` 和 `mcp_rag_server.py` 通过 `COPY` 打进镜像：

```dockerfile
COPY rag_api/api_agent.py ./rag_api/api_agent.py
COPY mcp_test/mcp_rag_server.py ./mcp_test/mcp_rag_server.py
COPY docs/ ./docs/
COPY faiss_index/ ./faiss_index/
```

每次改一行代码，都要：

```cmd
docker compose down
docker compose build      # 2~5 秒到几分钟
docker compose up -d
```

**磁盘越吃越多，迭代越来越慢。**

### 分析
**为什么 `COPY` 会导致 rebuild**：
- `COPY` 把文件**打进镜像层**
- 文件内容变了 → 这一层的 hash 变了 → 后续所有层缓存失效
- 必须重新 build

**磁盘为什么涨**：
- 每次 build 生成新层
- 旧层变"悬空镜像"（dangling image）
- 不像 Git，Docker 不会自动清理

**实测**：项目磁盘 `docker system df` 显示：

```
Images          7         3         28.31GB   17.06GB (60%)
Build Cache     22        0         11.01GB   0B
```

镜像可回收 17 GB，缓存 11 GB。折腾几轮就吃掉几十 GB。

### 方案
**开发期把频繁变动的目录改成 volume 挂载**：

```yaml
volumes:
  - ./hf_cache:/root/.cache/huggingface
  - ./data:/app/data
  # === 开发期挂载：改代码不用 rebuild ===
  - ./rag_api:/app/rag_api
  - ./mcp_test:/app/mcp_test
  - ./rag_core:/app/rag_core
  - ./docs:/app/docs
  - ./faiss_index:/app/faiss_index
```

**关键调整**：`DATA_DIR` 从 `/app/rag_api/data` 改成 `/app/data`。

因为 `./rag_api` 整个挂到 `/app/rag_api`，原来的 `./data:/app/rag_api/data` 会和它冲突。分开后互不影响。

**Dockerfile 不动**，`COPY` 保留作为"生产模式"的 fallback。

### 效果对比

| 改了什么 | 挂载前 | 挂载后 |
|---|---|---|
| `api_agent.py` | `down` + `build` + `up` | `docker compose restart agent-api` |
| `docs/` | `down` + `build` + `up` | restart |
| `faiss_index/` | `down` + `build` + `up` | restart |
| 时间 | 几秒到几分钟 | 2~3 秒 |
| 磁盘 | 每次涨 | 零增长 |

### 验证

改 `api_agent.py`：

```python
return {"message": "Agent API v2（挂载生效）"}
```

**不要 build**，直接：

```cmd
docker compose restart agent-api
curl http://127.0.0.1:8000/
{"message":"Agent API v2（挂载生效）"}
```

挂载生效。

### 收尾时改回 COPY

项目结束、要交付时，把开发挂载注释掉：

```yaml
volumes:
  - ./hf_cache:/root/.cache/huggingface
  - ./data:/app/data
  # === 开发期挂载，收尾时注释掉 ===
  # - ./rag_api:/app/rag_api
  # - ./mcp_test:/app/mcp_test
  # - ./rag_core:/app/rag_core
  # - ./docs:/app/docs
  # - ./faiss_index:/app/faiss_index
```

重新 build：

```cmd
docker compose down
docker compose build
docker compose up -d
```

镜像就是"冻结"的、自包含的，不依赖宿主机文件。

### 为什么这么设计

| 阶段 | 用哪种 | 原因 |
|---|---|---|
| 开发 | volume 挂载 | 迭代快，磁盘不涨 |
| 生产 | COPY 进镜像 | 可复现，不依赖宿主机文件 |

这是 Docker 的标准做法。**开发期追求迭代速度，生产期追求可复现性。**

### 收获
**Docker 是分层缓存，但每次内容变了都会新增一层。** 频繁改动的文件不该用 `COPY`，而应该挂载。

**开发用挂载，生产用 COPY。** 这不是"哪个更好"，而是"哪个适合当前阶段"。

**磁盘爆炸是渐进式的。** `docker system df` 一看就知道——镜像 28 GB、缓存 11 GB，大部分是历史版本。每次 build 都涨，清理要手动。

**关键调整：`DATA_DIR` 要跟着改。** 挂载 `./rag_api` 后，原来的 `./data:/app/rag_api/data` 冲突。改成 `/app/data` 就分开了。

**面试时能讲**：
> "开发期我把代码和语料目录挂载进容器，改代码只要 `docker compose restart`，不用 rebuild。因为项目早期每次改一行都要 build，磁盘也越吃越多——`docker system df` 一看，镜像 28 GB、缓存 11 GB。挂载后改代码 2 秒生效，磁盘零增长。收尾时把挂载注释掉重新 build，镜像就是自包含的，不依赖宿主机文件。开发期追求迭代速度，生产期追求可复现性。"

---

## 故事 34：限流从"预留"到"落地"（2026-10-14）

### 问题
Redis 依赖一直挂在 `requirements.txt` 里，但整个项目**没有任何地方用它**。故事 19 删掉了缓存，Redis 就成了纯"预留"。

面试官问"你为什么留着 Redis 依赖"，只能答"将来做限流"。**将来是多久？**

### 分析
**"预留依赖"是工程债的软性版本**：
- 不算 bug，但项目里躺着一个"从没用过的依赖"
- 面试时讲"我预留做限流"听起来像是"想做但没做"
- 真正落地了才能讲"我用 `incr + expire` 做限流"

**为什么限流必须做**：
- 接口已经加了鉴权（故事 32），但鉴权只防"谁能调"，不防"调多少次"
- 生产环境必须两层：**谁 + 多少**
- 没有限流 = 一个 Key 可以无限打，等于没有保护

### 方案

**① 固定窗口算法**

```python
def rate_limit(key: str):
    r = _get_redis()
    redis_key = f"rate:{key}"
    current = r.incr(redis_key)
    if current == 1:
        r.expire(redis_key, 60)   # 第一次访问设置 60 秒过期
    if current > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, detail="请求过于频繁")
```

**原理**：
- `incr` 原子自增，返回当前值
- 第一次（返回 1）时设过期时间，60 秒窗口
- 超过配额 → 429

**② 按 API Key 限流**

复用鉴权拿到的 token：

```python
async def verify_key(authorization: str = Header(default="")) -> str:
    ...
    return token

@app.post("/chat")
async def chat(payload: ChatRequest, token: str = Depends(verify_key)):
    rate_limit(token)
    ...
```

**③ fail-open**

```python
try:
    ...  # Redis 操作
except HTTPException:
    raise   # 限流本身触发，要抛出
except redis.RedisError as e:
    print(f"[限流] Redis 不可用，放行：{e}")
    # 不抛异常，继续执行主流程
```

**Redis 挂了放行，不是拒绝。** 限流是保护措施，不能让它成为新的单点故障。

### 验证

```cmd
for /L %i in (1,1,21) do @curl.exe ... -w "%i : %{http_code}\n" ...
```

输出：

```text
1 : 200
...
20 : 200
21 : 429
```

Redis 里：

```cmd
docker exec -it agent-redis redis-cli get "rate:wzyyqw106"
:: "21"
docker exec -it agent-redis redis-cli ttl "rate:wzyyqw106"
:: 47（剩余秒数）
```

### 关键设计

| 点 | 说明 |
|---|---|
| 维度 | 按 API Key，复用鉴权拿到的 token |
| 算法 | 固定窗口（incr + expire），不做滑动窗口 |
| fail-open | Redis 挂了放行，不能挂掉主流程 |
| 配额可配 | 环境变量 `RATE_LIMIT_PER_MINUTE` |
| `/history` 不限流 | 展示层流量小 |

### 收获
**"预留"变"落地"只需要 30 行代码。** 但差的是"我打算做"和"我做了"——面试里是两个量级的回答。

**fail-open 是生产级设计。** 加一个依赖前先想清楚：它挂了怎么办？限流、监控、日志这些"辅助功能"，挂了要放行，不能变成新的故障点。

**固定窗口够用就别做滑动窗口。** 项目规模小，20 次/分钟的精度要求不高。滑动窗口是"更精确但也更复杂"，不是"更好"。

**面试时能讲**：
> "我把 Redis 从'预留'变成'落地'——按 API Key 限流，20 次/分钟。用 `incr + expire` 做固定窗口，比滑动窗口简单，够用。关键设计是 **fail-open**：Redis 挂了就放行，限流是保护措施，不能让它成为新的单点故障。鉴权 + 限流是生产环境的两层保护：鉴权防'谁能调'，限流防'调多少次'。"

---

## 故事 35：从"能跑"到"有测试和 CI"（2026-10-16）

### 问题
功能全跑通了，但**一个自动化测试都没有**——所有验证都是手工的：起服务、curl、肉眼看输出。后果是：
- 改一行代码，不知道会不会碰坏别的地方
- 复现 bug 靠记忆，不靠脚本

### 分析
动手写测试时先撞上一个结构性问题：**`eval/` 和 `scripts/` 根本没法被 pytest 导入**——它们在 import 时就执行（加载模型、跑评估）。这不是"测试写得不好"，是**代码不可测**，属于设计问题。

另一个现实约束：检索测试要加载 bge 模型（几百 MB），不可能每次 push 都跑。

### 方案
1. **分两层**：单元测试（打桩、秒级）+ 集成测试（需模型，`@pytest.mark.integration` 标记）
2. **接口测试打桩**：假 Agent / 假 MySQL / 假 Redis，只断言"有没有按租户拼 thread_id""SQL 有没有带 owner 条件"，不依赖真实 LLM/DB
3. **降级测试用机制而非数据**：把 `max_distance` 调到 `-1.0` 确定性触发降级分支，而不是赌某条查询的距离刚好超阈值
4. **CI 只跑单元测试**：`.github/workflows/ci.yml`，push / PR 触发

结果：23 个用例，单元 2 秒、集成 23 秒。

### 收获
**可测试性是设计出来的。** 脚本要写成 `def main()` + `if __name__ == "__main__": main()`，才能被 import、被测试——不然它只是"能跑一次的脚本"。

**测试替身分清 Stub / Mock / Fake。** 这里用 **Fake**（有真实逻辑的替身，比如 `_FakeRedis` 真的会计数），比 Mock（只断言调用）更贴近真实行为。

**CI 是"作品"和"工程"的分界线。** 它的价值不是"跑测试"，而是**让回归在合并前暴露**——不靠人记得跑。

### 面试时能讲
> "项目一开始零自动化测试。加测试时发现 `eval/` 和 `scripts/` 在 import 时就执行、根本没法被测，这是可测试性的设计问题。我按测试金字塔分两层：单元测试打桩（假 Agent / 假 MySQL / 假 Redis），断言租户隔离和 SQL 条件；集成测试标记 `integration`，因为要下模型所以 CI 跳过。还踩了个坑：测'降级'不能靠一条查询赌它距离超阈值，要把阈值调到不可能满足来确定性触发分支。最后加了 GitHub Actions，push 就自动跑单元测试。"

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
| 构建优化 | Dockerfile 指令顺序影响缓存效率；多阶段构建瘦身；开发期用挂载 |
| 协议理解 | stdio 通信要用 stderr 打日志 |
| 检索算法 | 混合检索要调权重，MMR 是 chunk 级多样性 |
| 生产级意识 | Agent 要有护栏，记忆要持久化，接口要鉴权，限流要落地 |
| 功能演进回归 | 加新功能要检查旧设计的假设是否还成立 |
| 幻觉治理 | Prompt 软约束 + 检索层硬降级，两层防护 |
| 评估方法 | 单一指标会误导，要多个指标一起看 |
| 记忆持久化 | 让 checkpointer 管历史，不要在框架外面手动糊一层 |
| 架构重构 | 发现职责重叠就统一到单一真相源 |
| 安全防护 | 服务端日志记详细，客户端响应给友好提示；接口加鉴权 |
| 缓存正确性 | 缓存不是免费午餐，命中率结构性为 0 就删掉 |
| 依赖管理 | requirements.txt 必须经过干净环境验证 |
| **磁盘与 Docker 环境** | **Docker 构建失败先查磁盘空间；WSL2 vhdx 默认在 C 盘要迁走** |
| **依赖下载与平台兼容** | **CPU 版 torch 能砍 90% 下载量；容器内下载比宿主机稳** |
| **跨平台 wheel 陷阱** | **wheel 是平台相关的，宿主机给容器下 wheel 必须指定 `--platform`** |
| **模型缓存与容器** | **别硬编码 HF 缓存路径；三级探测让本地和容器都能跑** |
| **checkpoints 持久化** | **`docker compose down` 会丢记忆，必须挂 volume 到宿主机** |
| **评估代码分叉** | **任何被多消费方引用的逻辑都要抽公共模块；评估脚本必须能自己跑出所有报出的数字** |
| **阈值拍脑袋定 / 重标定** | **超参数要么标定，要么写清楚是拍的；标定脚本比阈值本身更有价值；语料扩展后要重标定** |
| **Rerank 收益被语料压制** | **精排有效，但小语料天花板低；扩语料才能发挥全部威力** |
| **测试集太小判别力不足** | **30 条样本 + 2 篇文档 → 随机基线 84.1%，判别空间 15.9 个点；扩到 249 条 + 98 篇 / 583 chunk → 随机基线 3.0%，判别空间 54.0 个点** |
| **接口裸奔** | **生产环境必须有鉴权；用 FastAPI 的 Depends 做依赖注入，鉴权逻辑和业务解耦** |
| **改代码每次 rebuild** | **开发期 volume 挂载，生产期 COPY 冻结；频繁变动的文件不该打进镜像** |
| **限流未落地** | **"预留"变"落地"只需 30 行；限流用 incr+expire 固定窗口；fail-open 避免引入新单点** |

**每个问题的解决过程都体现了同一个原则：先分析根本原因，再选择方案，最后量化效果。**