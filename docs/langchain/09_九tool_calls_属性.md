# 九、`tool_calls` 属性

### 代码

```python
if hasattr(msg, 'tool_calls') and msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"  → 调用工具: {tc['name']}, 参数: {tc['args']}")
```

### 解析

模型决定调用工具时，`AIMessage` 特点：
- `content` 为空（未生成最终回答）
- `tool_calls` 包含工具调用信息

`tc` 结构：

```python
{
    'name': 'get_weather',
    'args': {'city': '北京'},
    'id': 'call_00_xxx',
    'type': 'tool_call'
}
```
