# 一、`@tool` 装饰器

### 代码

```python
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}今天天气晴朗，气温 25℃，适合出门。"
```

### 解析

`@tool` 是 Python 装饰器，把普通函数包装成 LangChain 的 `StructuredTool` 对象。

它自动提取三样东西转成 JSON 发给模型：

| 提取内容 | 来源 | 作用 |
|---|---|---|
| 工具名 | 函数名 `get_weather` | 模型决定调哪个工具 |
| 工具描述 | docstring | 模型判断工具用途 |
| 参数结构 | 类型注解 `city: str` | 生成 JSON Schema |

发给模型的 JSON 示例：

```json
{
  "name": "get_weather",
  "description": "获取指定城市的天气信息",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  }
}
```

**关键点**：
- 没有 docstring，模型可能不知道工具用途
- 没有类型注解，模型不知道参数格式
- `@tool` 可省略，但加上更显式
