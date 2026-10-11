# 故事 20：requirements.txt 从未在干净环境验证

### 问题
项目里 `pip install` 装了一堆包，但 `requirements.txt` 只写了最初那 10 个。

在干净 venv 里跑 `pip install -r requirements.txt` 后，`python -c "import rag_api.api_agent"` 立刻报错：
```text
ModuleNotFoundError: No module named 'redis'
```

### 分析
代码里实际 import 的第三方包有 18 个，`requirements.txt` 里只声明了 10 个。**缺 8 个**：

| 缺失包 | 被谁 import |
|---|---|
| `redis` | api_agent.py |
| `pymysql` | api_agent.py |
| `langchain-community` | mcp_rag_server.py（FAISS） |
| `langchain-text-splitters` | mcp_rag_server.py（切分） |
| `langgraph-checkpoint-sqlite` | api_agent.py（AsyncSqliteSaver） |
| `aiosqlite` | AsyncSqliteSaver 的底层依赖 |
| `jieba` | mcp_rag_server.py（中文分词） |
| `rank-bm25` | mcp_rag_server.py（BM25） |

**根因**：开发过程中逐个 `pip install` 补包，**忘了同步回 `requirements.txt`**。

### 杀伤力
- 面试官 clone 下来，`pip install -r requirements.txt` 装到一半就报错
- Docker 构建时，容器里缺 `redis`，`uvicorn` 一 import 就崩
- CI 流水线同理

**这是"项目能不能在别人机器上跑"的最底线问题。**

### 方案
1. 遍历项目里所有 `.py` 文件，提取所有 `import` 语句
2. 映射到对应的 pip 包名（注意：import 名里的下划线 `_` 通常要换成连字符 `-`）
3. 补全 `requirements.txt`
4. **在干净 venv 里验证**：
   ```cmd
   python -m venv .venv-test
   .venv-test\Scripts\activate
   pip install -r requirements.txt
   python -c "import rag_api.api_agent; print('导入成功')"
   ```

### 修复后的清单（18 个）

```text
fastapi>=0.136.0
uvicorn>=0.48.0
langchain>=1.4.0
langchain-openai>=1.6.0
langchain-mcp-adapters>=0.3.0
langchain-community>=0.4.0
langchain-text-splitters>=1.0.0
langgraph>=1.2.0
langgraph-checkpoint-sqlite>=2.0.0
aiosqlite>=0.20.0
fastmcp>=3.4.0,<4.0.0
sentence-transformers>=5.5.0
faiss-cpu>=1.14.0
python-dotenv>=1.2.0
pymysql>=1.1.0
jieba>=0.42.0
rank-bm25>=0.2.2
```

（`redis` 暂时保留，预留做限流）

### 为什么这个问题容易被忽略
- 开发时，**你机器上碰巧有**这些包（之前手动装过）
- 所以你本地能跑，感觉不到问题
- 但换机器就崩

**依赖清单必须经过干净环境验证，否则就是装饰品。**

### 收获
**每装一个新包，立刻 `pip freeze` 或手动补进 `requirements.txt`。** 更好的做法是用 `uv` 或 `poetry`，它们在装包时自动维护依赖清单，不需要手动同步。

**面试时能讲**：
> "我之前踩过一个坑：项目里 `pip install` 装了一堆包，但 `requirements.txt` 只写了最初那 10 个。后来在一个干净 venv 里跑 `pip install -r requirements.txt`，立刻发现 `redis`、`pymysql`、`jieba`、`rank-bm25` 这些都没声明。修法是把所有 `.py` 文件的 import 语句扫一遍，映射到 pip 包名，补全清单，再在新环境里验证。这件事让我意识到：依赖清单必须经过干净环境验证，否则就是装饰品。"

---
