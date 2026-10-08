# RAG Agent 服务

基于 LangChain + LangGraph + MCP + RAG 的企业级 Agent 服务，支持 SSE 流式输出，持久化记忆，Docker 容器化部署。

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
                    降级判断（距离阈值）
                            ↓ 通过
                    混合检索（向量 + BM25 + 加权 RRF）
                            ↓
                        FAISS 向量库
                            ↓ 检索结果返回给 Agent
                        DeepSeek 生成回答
    ↓ SSE 流式返回
客户端

记忆层：
- LangGraph checkpointer（唯一真相源）
- AsyncSqliteSaver 持久化到 SQLite 文件（重启不丢）
- checkpoints.db 挂载到宿主机 ./data/，容器重建也不丢
- MySQL 只做展示层（给 /history 接口用）

缓存层：
- 暂时没有缓存。历史版本用过 Redis，key 格式 chat:{thread_id}:v{context_version}:{md5}，
  但实测发现 context_version 每轮 +2，命中率结构性为 0，属于负优化，已删除。
- Redis 依赖保留在 requirements.txt 中，预留做限流（incr + expire）。
```

## 核心功能

- **持久化记忆**：LangGraph checkpointer + AsyncSqliteSaver，服务重启不丢历史
- **RAG 检索增强**：本地 embedding（BAAI/bge-small-zh-v1.5）+ FAISS 持久化
- **混合检索**：向量 + BM25 + 加权 RRF（向量 0.7 / BM25 0.3）
- **降级策略**：向量距离阈值，检索不到就不调 LLM，避免幻觉
- **MCP 工具解耦**：工具独立成 Server，跨进程 stdio 通信
- **SSE 流式输出**：逐 token 返回，中间步骤过滤
- **MySQL 展示层**：对话历史按 thread_id 持久化，供 `/history` 接口查询
- **Agent 护栏**：`recursion_limit=10` + `max_tokens=2000`，防止死循环和 Token 爆炸
- **友好错误处理**：异常映射为可读提示，不泄漏 API key、路径等内部信息
- **RAG 评估**：30 个测试项，四指标对比（召回率 / 精确率 / MRR / 多样性）
- **Docker 部署**：多阶段构建，镜像 2.42 GB，一键启动

## 目录结构

| 文件夹 | 内容 |
|---|---|
| `rag_api/` | FastAPI + Agent 服务（核心） |
| `mcp_test/mcp_rag_server.py` | RAG MCP Server |
| `docs/` | 知识库文档（真实笔记） |
| `eval/` | RAG 评估脚本 + 测试集 |
| `faiss_index/` | FAISS 索引（204 KB + 44 KB，已提交进 Git） |
| `hf_cache/` | HuggingFace 模型缓存（volume 挂载进容器，不提交） |
| `data/` | checkpoints.db 持久化目录（volume 挂载，不提交） |
| `first_test/` | 早期手写 Agent 练习 |
| `langchain_test/` | LangChain 组件学习 |
| `langgraph_test/` | LangGraph 状态图学习 |
| `stream_test/` | 流式输出实验 |

## 启动

### 本地开发

```bash
# 1. 启动 MySQL（Redis 可选）
docker compose -f docker-compose.db.yml up -d

# 2. 启动 API 服务
cd rag_api
uvicorn api_agent:app --reload
```

访问 http://127.0.0.1:8000/docs

### Docker 部署

```bash
# 1. 首次使用前：本地下载 embedding 模型到 hf_cache/
#    （模型 192 MB，避免容器内首次启动时再下载）
#    注意：必须指定 cache_dir，否则默认写到 ~/.cache/huggingface，
#    容器 volume 挂载的 ./hf_cache/ 会是空的
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='hf_cache/hub')"

# 2. 构建并启动（首次约 5~10 分钟）
docker compose up --build -d

# 3. 后续启动（镜像已存在，几秒）
docker compose up -d

# 4. 查看日志
docker compose logs -f agent-api

# 5. 停止
docker compose down
```

### Docker 磁盘优化建议

Docker Desktop 在 Windows 上默认把 WSL2 虚拟磁盘存在 C 盘，长期使用会撑爆系统盘。**建议首次装完 Docker 就改**：

- Settings → Resources → Advanced → **Disk image location** 改成 `D:\Docker`（或空间充足的盘）

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 健康检查 |
| POST | `/chat` | 普通对话 |
| POST | `/chat/stream` | 流式对话（SSE） |
| GET | `/history` | 查看对话历史（MySQL 展示层） |

### 请求示例

```bash
# 普通对话（带 thread_id 区分会话）
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "MCP 是什么？", "thread_id": "user_001"}'

# 流式对话
curl -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "MCP 是什么？", "thread_id": "user_001"}'

# 查看历史
curl "http://127.0.0.1:8000/history?thread_id=user_001"
```

### 多轮对话演示

```bash
# 第一轮
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "MMR 是什么？", "thread_id": "test_001"}'

# 第二轮（用代词指代）
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "它和普通相似度检索有什么区别？", "thread_id": "test_001"}'

# 重启服务后再问，仍能理解"它"指 MMR
```

## 环境变量

复制 `.env.example` 为 `.env`，填写：

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
MYSQL_PASSWORD=your_mysql_password
```

## RAG 评估

构建 30 个测试项，四个指标对比：

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 100% | **0.87** | 0.96 | 1.37 |
| MMR | 100% | 0.83 | **0.97** | 1.43 |
| MMR + 按来源去重 | 100% | 0.77 | **0.97** | **1.57** |

**关键发现**：
- 精确率和多样性呈 trade-off：追求精确用相似度，需要多角度用 MMR+去重
- MRR 三种策略都在 0.96 以上，Top-1 都很准
- 早期实验发现等权 RRF 反而比纯向量差（97%），加权后（向量 0.7 / BM25 0.3）恢复到 100%
- 随机基线召回率 84.1%，说明当前 30 条样本的判别力有限（差 15.9 个点）

运行评估：

```bash
cd eval
python eval_retrieval.py
```

## 技术栈

- **Agent 编排**：LangChain 1.4 + LangGraph 1.2
- **模型**：DeepSeek（OpenAI 兼容协议）
- **工具协议**：MCP（FastMCP 3.4.8）
- **检索**：sentence-transformers + FAISS + BM25 + 加权 RRF
- **降级**：向量距离阈值
- **记忆持久化**：AsyncSqliteSaver（生产可换 PostgresSaver）
- **展示层**：MySQL
- **服务**：FastAPI + Uvicorn
- **部署**：Docker + Docker Compose（多阶段构建，2.42 GB）
- **流式**：SSE

## 设计决策

| 决策 | 原因 |
|---|---|
| 记忆统一到 LangGraph checkpointer | 避免 InMemorySaver + MySQL 双源打架 |
| 删掉 Redis 缓存 | 命中率结构性为 0，是负优化 |
| 检索层降级（距离阈值） | 比 Prompt 软约束更可靠，从根源避免幻觉 |
| 错误映射为友好提示 | `str(e)` 可能泄漏 API key、路径等内部信息 |
| 混合检索加权 RRF | 等权 RRF 实测反而比纯向量差（97%） |
| Docker 多阶段构建 | builder 阶段装编译器，运行时不带，镜像从 3.51 GB 降到 2.42 GB |
| CPU 版 torch | 项目只用 bge-small 做 embedding，CPU 毫秒级；下载量从 5 GB 降到 600 MB |
| 模型缓存 volume 挂载 | 模型下载移出构建期，避免网络超时；重建不重下 |
| FAISS 索引提交进 Git | 只有 248 KB，别人 clone 后可直接构建 |

## 已知不足

- pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环（生产应换 `aiomysql`）
- `save_conversation` 每次请求新建 MySQL 连接，无连接池
- `MAX_DISTANCE = 1.1` 阈值校准不够系统，应该对正负样本统计距离分布取分离点
- 评估是**文档级召回**而非 chunk 级，样本 30 条偏少，且缺端到端指标
- 流式状态机用布尔变量表达，边界情况可能出错
- 未实现 Rerank 精排

## 学习路径

ReAct → LangChain → LangGraph → RAG → MCP → FastAPI → Docker