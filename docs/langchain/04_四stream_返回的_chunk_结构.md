# 四、`stream` 返回的 `chunk` 结构

### 代码

```python
for chunk in agent.stream(...):
    for node_name, node_data in chunk.items():
        for msg in node_data.get("messages", []):
```

### 解析

每次 `chunk` 示例：

```python
{
    "model": {
        "messages": [AIMessage(content='', tool_calls=[...])]
    }
}
```

- 外层 key `"model"`：节点名
- 内层 value `{"messages": [...]}`：该节点的状态更新
- `node_data.get("messages", [])`：安全取消息列表
