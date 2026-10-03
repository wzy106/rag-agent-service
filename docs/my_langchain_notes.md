# LangChain 代码逐行解析笔记

> 用途：理解已跑通的代码，每个知识点对应项目里的真实代码。

## 一、`@tool` 装饰器

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

## 二、`create_agent` 的参数

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

## 三、`config` 和 `thread_id`

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

## 四、`stream` 返回的 `chunk` 结构

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

## 五、`prompt | model | parser` 管道语法

### 代码

```python
chain = prompt | model | parser
result = chain.invoke({"query": "北京今天天气怎么样？"})
```

### 解析

Python 通过运算符重载实现 `|`，称为 LCEL。

执行流程：

```text
prompt.invoke({"query": "..."})
    ↓ ChatPromptValue
model.invoke(ChatPromptValue)
    ↓ AIMessage（JSON 字符串）
parser.invoke(AIMessage)
    ↓ WeatherInfo 对象（Pydantic）
```

等价于：

```python
result = parser.invoke(model.invoke(prompt.invoke({"query": "..."})))
```

## 六、`PydanticOutputParser` 结构化输出

### 代码

```python
class WeatherInfo(BaseModel):
    city: str = Field(description="城市名称")
    weather: str = Field(description="天气状况，如晴、多云")
    temperature: int = Field(description="温度，摄氏度")

parser = PydanticOutputParser(pydantic_object=WeatherInfo)
prompt = ChatPromptTemplate.from_messages([
    ("system", "请严格按格式输出，只输出JSON。\n{format_instructions}"),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
result = chain.invoke({"query": "北京今天天气怎么样？"})
```

### 解析

**为何不用 `with_structured_output`？**
DeepSeek 不支持 `json_schema` 模式。

**原理**：
1. `parser.get_format_instructions()` 生成格式说明
2. 通过 `.partial()` 注入 system prompt
3. 模型输出 JSON 字符串
4. `parser` 本地解析为 Pydantic 对象

## 七、`InMemorySaver` 记忆机制

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

## 八、`System Prompt` 人设控制

### 代码

```python
agent = create_agent(
    model,
    tools=[get_weather, calculator],
    checkpointer=memory,
    system_prompt="你是一个甜妹专业解说员，回答简洁，不要超过30个字"
)
```

### 解析

`system_prompt` 是放在对话最前面的 System Message，控制模型行为：
- 设定角色、回答长度、格式、语气
- 不是硬性规则，模型可能偶尔不遵守
- 不听话时可加强措辞或用中间件强制约束

## 九、`tool_calls` 属性

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

## 十、并行工具调用

### 输出示例

```text
=== 节点: model ===
  [ai] I'll help you with both questions.
    → 调用工具: get_weather, 参数: {'city': '北京'}
    → 调用工具: calculator, 参数: {'a': 123, 'b': 456}
```

### 解析

模型一次性发起多个工具调用，称为并行工具调用。

**触发条件**：
- 用户问题包含多个独立任务
- 任务间无依赖
- 工具定义清晰

## 十一、常见错误对照表

| 错误 | 原因 | 解决 |
|---|---|---|
| `NameError: name 'result' is not defined` | 用了 `stream` 却访问 `result` | 用 `invoke` 才有 `result`；`stream` 遍历 `chunk` |
| `SyntaxError: invalid character '”'` | 中文引号与英文引号混用 | 字符串边界用英文 `"` |
| `This response_format type is unavailable` | DeepSeek 不支持 `json_schema` | 改用 `PydanticOutputParser` |
| `OPENAI_API_KEY=sk-xxx` 报错 | 参数名错误 | 应为 `api_key="sk-xxx"` |
| 工具不被调用 | 缺 docstring 或描述不清 | 给每个 `@tool` 函数写清楚 docstring |

## 十二、学习方式总结

```text
拿到代码
    ↓
先跑通
    ↓
逐行问"这行干嘛的"
    ↓
理解每个参数、属性、流程
    ↓
改一个参数，看结果变不变
    ↓
才算真的会了
```

**跑通 ≠ 理解。** 能解释每一行代码，才是真正掌握。