# 六、`PydanticOutputParser` 结构化输出

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
