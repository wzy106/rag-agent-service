# 故事 8：MCP Server 里的 print 看不到

### 问题
在 `mcp_rag_server.py` 里加 `print("[RAG Server] 索引构建完成")`，跑起来完全看不到输出。

### 分析
MCP 用 stdio 通信，`stdout` 被协议占用。`print` 默认输出到 `stdout`，被协议管道拦截了。

### 方案
改成输出到 `stderr`：
```python
import sys
print("[RAG Server] 索引构建完成", file=sys.stderr)
```

### 为什么这么选
- **stderr 不被占用**：MCP 只用 stdout 通信，stderr 是自由的
- **改一行就行**：不用重构日志系统

### 收获
使用 stdio 通信的协议，日志必须走 stderr，否则会被协议吞掉。这是理解 MCP 底层机制的一个切入点。

---
