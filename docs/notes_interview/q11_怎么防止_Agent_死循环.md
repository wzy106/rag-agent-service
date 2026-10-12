# 11. 怎么防止 Agent 死循环？

**两道护栏**：

1. **`recursion_limit=10`**：Agent 最多循环 10 次，超过抛异常
2. **`max_tokens=2000`**：单次输出最多 2000 token

**触发后怎么办**：异常被 `try/except` 捕获，以 SSE 格式返回友好错误，服务不崩。

**面试话术**：
> "我给 Agent 设了两道护栏：一是 recursion_limit=10，超过自动抛异常终止；二是 max_tokens=2000，限制单次输出。第一道防止 Agent 无限调用工具，第二道防止单次回答过长。"
