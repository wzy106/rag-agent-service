# 五、`prompt | model | parser` 管道语法

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
