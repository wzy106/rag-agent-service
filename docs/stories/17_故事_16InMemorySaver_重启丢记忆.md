# 故事 16：InMemorySaver 重启丢记忆

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
