# 故事 11：Agent 可能死循环或 Token 爆炸

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
