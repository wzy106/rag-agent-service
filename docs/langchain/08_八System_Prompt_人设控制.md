# 八、`System Prompt` 人设控制

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
