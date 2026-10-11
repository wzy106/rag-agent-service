# 故事 1：DeepSeek 不支持 json_schema

### 问题
做结构化输出时，用 LangChain 的 `with_structured_output` 报错：
```text
Error code: 400 - This response_format type is unavailable now
```

### 分析
LangChain 默认用 `json_schema` 模式强制模型输出结构化数据，但 DeepSeek 的 API 不支持这个模式。

### 方案
改用 `PydanticOutputParser`：
1. `parser.get_format_instructions()` 生成格式说明
2. 通过 `.partial()` 注入 system prompt
3. 模型输出 JSON 字符串
4. `parser` 本地解析为 Pydantic 对象

### 为什么这么选
- **兼容性最好**：不依赖模型 API 对 `json_schema` 的支持
- **原理透明**：提示词工程 + 本地解析，完全可控
- **可切换**：如果将来换模型支持 `json_schema`，可以随时切回

### 收获
模型 API 的差异需要用适配层来屏蔽，不能假设所有模型都支持相同特性。

---
