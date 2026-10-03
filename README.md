# AI Agent 学习项目

从零学习 AI Agent 开发的完整代码库，覆盖 ReAct、LangChain、LangGraph、RAG、MCP、FastAPI、Docker。

## 学习路径

ReAct → LangChain → LangGraph → RAG → MCP → FastAPI → Docker

## 目录结构

| 文件夹 | 内容 |
|---|---|
| `first_test/` | 最早的手写 Agent 练习 |
| `langchain_test/` | LangChain 核心组件：工具、记忆、结构化输出 |
| `langgraph_test/` | LangGraph 状态图、条件边、Checkpointer |
| `stream_test/` | 流式输出实验 |
| `mcp_test/` | MCP 最小示例 + RAG MCP Server |
| `rag_api/` | FastAPI + RAG Agent 服务（核心项目） |

| 根目录文件 | 内容 |
|---|---|
| `Dockerfile` | Docker 镜像构建 |
| `docker-compose.yml` | 容器编排配置 |
| `requirements.txt` | Python 依赖清单 |
| `.env` | 环境变量（不提交 Git） |

## 核心项目：RAG Agent 服务

### 架构

客户端 → FastAPI → LangChain Agent → MCP Client → MCP Server → 向量库

### 功能

- 本地 embedding（BAAI/bge-small-zh-v1.5）
- 内存向量库检索
- DeepSeek 大模型
- MCP 协议解耦工具
- 流式输出（SSE）
- 异常处理

### 启动（本地开发）

```bash
cd rag_api
uvicorn api_agent:app --reload
```

访问 http://127.0.0.1:8000/docs

### 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 健康检查 |
| POST | `/chat` | 普通对话 |
| POST | `/chat/stream` | 流式对话（SSE） |

### 环境变量

`.env`：

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### Docker 部署

```bash
# 构建并启动（第一次，约 15 分钟）
docker compose up --build -d

# 后续启动（镜像已存在，几秒）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

访问 http://127.0.0.1:8000/docs

## 技术栈

- LangChain 1.4.2
- LangGraph 1.2.12
- FastAPI 0.136.3
- FastMCP 3.4.7
- sentence-transformers 5.5.1
- faiss-cpu 1.14.2
- DeepSeek API
- Docker + Docker Compose