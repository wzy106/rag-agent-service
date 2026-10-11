# 七、`InMemorySaver` 记忆机制

### 代码

```python
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
agent = create_agent(model, tools=[...], checkpointer=memory)
```

### 解析

`InMemorySaver` 是 LangGraph 的检查点保存器：
- 存内存（RAM），程序关闭即丢失
- 适合开发测试；生产建议 `PostgresSaver`

三个概念配合：

| 概念 | 作用 |
|---|---|
| `InMemorySaver` | 存记忆的地方 |
| `checkpointer=memory` | 把存储器接到 Agent |
| `thread_id` | 会话 ID，区分不同对话 |
