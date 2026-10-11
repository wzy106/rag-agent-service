# 故事 13：InMemorySaver 的局限

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
