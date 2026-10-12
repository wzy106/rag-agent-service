# 十二、FastAPI 服务层设计（面试重点）

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
| Agent 多轮对话怎么实现？ | LangGraph checkpointer + AsyncSqliteSaver + DATA_DIR |
| 容器重建后记忆怎么不丢？ | checkpoints volume 挂载到宿主机 ./data/ |
| 错误信息怎么处理？ | `friendly_error` 映射，用异常类型判断 |
| 检索不到内容怎么处理？ | 距离阈值降级（标定 0.96），返回“没找到”，不调 LLM |
| RAG 评估用什么指标？ | 召回率 + 精确率 + MRR + 多样性 + 随机基线 |
| 评估怎么保证可复现？ | 抽公共 `rag_core/search.py`，一条命令跑出所有数字 |
| 阈值怎么定？ | 正负样本距离分布 + PR 曲线标定（100 正 + 25 负） |
| 为什么要 Rerank？ | 双塔召回粗，交叉编码器精排 |
