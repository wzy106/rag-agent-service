# 12. Agent 多轮对话怎么实现？

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
1. `test_001` 问“MMR 是什么？” → 正常回答
2. **`docker compose down` 完全销毁容器**
3. `docker compose up -d` 重建
4. `test_001` 问“它和普通相似度检索有什么区别？” → 仍能理解“它”指 MMR ✅

**面试话术**：
> “我的记忆用 LangGraph 的 checkpointer 做唯一真相源，持久化用 AsyncSqliteSaver（生产环境会换 PostgresSaver）。每次请求只传当前这一轮的消息，Agent 自动从 checkpointer 加载该 thread_id 的历史。之前我试过手动从 MySQL 加载历史拼进 messages，但发现和 checkpointer 的 state 会打架——消息格式混用、历史可能重复累积。后来统一到 checkpointer，MySQL 只做展示层给 /history 接口用。另外我把 checkpoints.db 挂到宿主机的 ./data/，这样 docker compose down 后容器重建也不丢。”
