# RAG Agent 服务

基于 LangChain + LangGraph + MCP + RAG 的企业级 Agent 服务，支持 SSE 流式输出，Redis 缓存，MySQL 历史持久化，Docker 容器化部署。

## 架构

```text
客户端
    ↓ HTTP POST /chat/stream
FastAPI（api_agent.py）
    ↓ agent.astream()
LangChain Agent（ReAct 循环）
    ↓ 工具调用
MCP Client ──stdio──→ MCP Server（mcp_rag_server.py）
                            ↓
                        FAISS 向量库
                            ↓ 检索结果返回给 Agent
                        DeepSeek 生成回答
    ↓ SSE 流式返回
客户端
```

## 核心功能

- **RAG 检索增强**：本地 embedding（BAAI/bge-small-zh-v1.5）+ FAISS 持久化 + MMR 检索
- **MCP 工具解耦**：工具独立成 Server，跨进程 stdio 通信
- **SSE 流式输出**：逐 token 返回，中间步骤过滤
- **Redis 缓存**：高频问题缓存，命中不调大模型
- **MySQL 历史**：对话按 thread_id 持久化
- **RAG 评估**：30 个测试项，对比三种检索策略
- **异常处理**：错误以 SSE 格式返回，不返回 500 堆栈
- **Docker 部署**：一键启动
- **Agent 护栏**:（recursion_limit=10，max_tokens=2000）

## 目录结构

| 文件夹 | 内容 |
|---|---|
| `rag_api/` | FastAPI + Agent 服务（核心） |
| `mcp_test/mcp_rag_server.py` | RAG MCP Server |
| `docs/` | 知识库文档（真实笔记） |
| `eval/` | RAG 评估脚本 + 测试集 |
| `faiss_index/` | FAISS 索引（运行时生成） |
| `first_test/` | 早期手写 Agent 练习 |
| `langchain_test/` | LangChain 组件学习 |
| `langgraph_test/` | LangGraph 状态图学习 |
| `stream_test/` | 流式输出实验 |

## 启动

### 本地开发

```bash
# 1. 启动 MySQL 和 Redis
docker compose -f docker-compose.db.yml up -d

# 2. 启动 API 服务
cd rag_api
uvicorn api_agent:app --reload
```

访问 http://127.0.0.1:8000/docs

### Docker 部署

```bash
# 构建并启动（首次约 15 分钟）
docker compose up --build -d

# 后续启动（镜像已存在）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 健康检查 |
| POST | `/chat` | 普通对话（带 Redis 缓存） |
| POST | `/chat/stream` | 流式对话（SSE） |
| GET | `/history` | 查看对话历史（MySQL） |

### 请求示例

```bash
# 普通对话
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "MCP 是什么？"}'

# 流式对话
curl -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "MCP 是什么？"}'

# 查看历史
curl "http://127.0.0.1:8000/history?thread_id=user_001"
```

## 环境变量

复制 `.env.example` 为 `.env`，填写：

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
MYSQL_PASSWORD=your_mysql_password
```

## RAG 评估

构建 30 个测试项，对比三种检索策略：

| 策略 | 召回率 | 平均多样性 |
|---|---|---|
| 普通相似度 | 97% | 2.00 |
| MMR | 93% | 2.50 |
| MMR + 按来源去重 | — | 3.00 |

MMR 用 4% 的召回率换取 25% 的多样性提升，适合需要多角度信息的场景。

失败案例分析：用户口语化表达与文档术语存在语义鸿沟，企业级方案是混合检索 + Rerank。

运行评估：

```bash
cd eval
python eval_retrieval.py
```

## 技术栈

- **Agent 编排**：LangChain 1.4 + LangGraph 1.2
- **模型**：DeepSeek（OpenAI 兼容协议）
- **工具协议**：MCP（FastMCP 3.4.7）
- **检索**：sentence-transformers + FAISS + MMR
- **缓存**：Redis
- **历史**：MySQL
- **服务**：FastAPI + Uvicorn
- **部署**：Docker + Docker Compose
- **流式**：SSE

## 学习路径

ReAct → LangChain → LangGraph → RAG → MCP → FastAPI → Docker