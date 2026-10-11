# 二、`create_agent` 的参数

### 代码

```python
memory = InMemorySaver()
agent = create_agent(
    model,
    tools=[get_weather, calculator],
    checkpointer=memory,
    system_prompt="你是一个甜妹专业解说员，回答简洁，不要超过30个字"
)
```

### 解析

| 参数 | 作用 |
|---|---|
| `model` | 用哪个大模型思考与生成 |
| `tools` | 模型可调用的工具列表 |
| `checkpointer` | 记忆存储位置（不传则无记忆） |
| `system_prompt` | 人设指令，放在每条对话最前面 |

**本质**：`create_agent` 底层搭建了 LangGraph 状态图，循环为：

```text
调模型 → 判断调工具 → 执行工具 → 再调模型 → 最终回答
```
