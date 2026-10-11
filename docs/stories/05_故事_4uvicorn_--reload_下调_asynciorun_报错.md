# 故事 4：uvicorn --reload 下调 asyncio.run 报错

### 问题
在 FastAPI 模块顶层调用 `asyncio.run(load_tools())` 加载 MCP 工具，报错：
```text
RuntimeError: asyncio.run() cannot be called from a running event loop
```

### 分析
- uvicorn 启动后已经在一个事件循环里
- 模块顶层代码在 import 时执行，此时事件循环正在运行
- `asyncio.run()` 不能嵌套调用

### 方案
改用 FastAPI 的 `lifespan` 生命周期钩子：
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    client = MultiServerMCPClient({...})
    tools = await client.get_tools()
    agent = create_agent(model, tools=tools)
    yield
    print("[关闭] 服务停止")

app = FastAPI(lifespan=lifespan)
```

### 为什么这么选
- **官方推荐**：lifespan 是 FastAPI 替代废弃的 `on_event` 的标准方案
- **异步友好**：在事件循环里执行，可以 `await`
- **生命周期清晰**：yield 之前是启动逻辑，之后是关闭逻辑

### 收获
异步框架下，初始化逻辑应该放在生命周期钩子里，而不是模块顶层。

---
