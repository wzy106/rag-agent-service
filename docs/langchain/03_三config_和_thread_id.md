# 三、`config` 和 `thread_id`

### 代码

```python
config = {"configurable": {"thread_id": "user_001"}}
```

### 解析

LangGraph 规定的嵌套字典格式：
- 外层 `{"configurable": {...}}`：固定格式
- 内层 `{"thread_id": "user_001"}`：会话 ID

`thread_id` 存在 `checkpointer` 里。`InMemorySaver` 内部结构：

```python
{
    "user_001": [第一轮消息, 第二轮消息, ...],
    "user_002": [另一个用户的消息...],
}
```
