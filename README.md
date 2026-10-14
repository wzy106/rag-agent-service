# RAG Agent 服务

基于 LangChain + LangGraph + MCP + RAG 的企业级 Agent 服务，支持 SSE 流式输出，持久化记忆，API Key 鉴权，Redis 限流，Docker 容器化部署。

## 架构

```text
客户端
    ↓ HTTP POST /chat/stream  （带 Bearer token）
FastAPI（api_agent.py）
    ↓ verify_key 鉴权通过
    ↓ rate_limit 限流检查
    ↓ agent.astream()
LangChain Agent（ReAct 循环）
    ↓ 工具调用
MCP Client ──stdio──→ MCP Server（mcp_rag_server.py）
                            ↓
                    rag_core/search.py（公共混合检索）
                            ↓
                    降级判断（距离阈值 0.96）
                            ↓ 通过
                    混合检索召回 Top-20（向量 + BM25 + 加权 RRF）
                            ↓
                    Rerank 精排 Top-3（bge-reranker-base, CrossEncoder）
                            ↓
                    FAISS 向量库（98 篇文档 / 583 chunk）
                            ↓ 检索结果返回给 Agent
                        DeepSeek 生成回答
    ↓ SSE 流式返回
客户端

鉴权层：
- API Key 通过 Bearer token 传入
- Header + Depends 做依赖注入，鉴权逻辑和业务逻辑解耦
- API_KEY 从环境变量读，不写死在代码里

限流层：
- Redis 固定窗口限流（incr + expire）
- 按 API Key 限流，默认 20 次/分钟
- Redis 不可用时放行（fail-open），限流不能成为新的单点故障

记忆层：
- LangGraph checkpointer（唯一真相源）
- AsyncSqliteSaver 持久化到 SQLite 文件（重启不丢）
- checkpoints.db 挂载到宿主机 ./data/，容器重建也不丢
- MySQL 只做展示层（给 /history 接口用）

缓存层：
- 暂时没有缓存。历史版本用过 Redis 缓存，key 格式 chat:{thread_id}:v{context_version}:{md5}，
  但实测发现 context_version 每轮 +2，命中率结构性为 0，属于负优化，已删除。
```

## 核心功能

- **持久化记忆**：LangGraph checkpointer + AsyncSqliteSaver，服务重启/容器重建不丢历史
- **API Key 鉴权**：`/chat` 和 `/chat/stream` 需要 Bearer token；`/history` 和 `/` 保持公开
- **Redis 限流**：按 API Key 限流，20 次/分钟（`incr + expire` 固定窗口，fail-open）
- **RAG 检索增强**：本地 embedding（BAAI/bge-small-zh-v1.5）+ FAISS 持久化
- **两阶段检索**：混合检索召回 Top-20 → Rerank 精排 Top-3
- **混合检索**：向量 + BM25 + 加权 RRF（向量 0.7 / BM25 0.3）
- **Rerank 精排**：bge-reranker-base（CrossEncoder），召回率 +4.4%，MRR +0.05
- **公共检索模块**：`rag_core/search.py`，服务端和评估脚本共用同一份 `hybrid_search`
- **降级策略**：向量距离阈值（标定为 0.96），检索不到就不调 LLM，避免幻觉
- **MCP 工具解耦**：工具独立成 Server，跨进程 stdio 通信
- **SSE 流式输出**：逐 token 返回，中间步骤过滤
- **MySQL 展示层**：对话历史按 thread_id 持久化，供 `/history` 接口查询
- **Agent 护栏**：`recursion_limit=10` + `max_tokens=2000`，防止死循环和 Token 爆炸
- **友好错误处理**：异常映射为可读提示，不泄漏 API key、路径等内部信息
- **RAG 评估**：249 条测试项，5 种策略对比 + 随机基线（3.0%），一条命令可复现
- **测试集自动生成**：脚本用 DeepSeek API 批量生成 249 条
- **阈值标定**：正负样本距离分布 + PR 曲线，精确率 99.0%、召回率 100%
- **Docker 部署**：多阶段构建，镜像 2.42 GB，一键启动

## 目录结构

| 文件夹 | 内容 |
|---|---|
| `rag_api/` | FastAPI + Agent 服务（核心） |
| `rag_core/search.py` | 公共检索模块（含 Rerank，服务端 + 评估脚本共用） |
| `mcp_test/mcp_rag_server.py` | RAG MCP Server |
| `docs/` | 知识库文档（98 篇 markdown，按主题分 5 个子目录） |
| `docs/langchain/` | LangChain 组件教程（12 篇） |
| `docs/notes/` | 项目笔记（15 篇） |
| `docs/notes_interview/` | 面试题逐个拆分（25 篇） |
| `docs/readme/` | README 按章节拆分（13 篇） |
| `docs/stories/` | 踩坑故事（33 篇） |
| `eval/` | RAG 评估脚本 + 测试集（249 条）+ 阈值标定 |
| `eval/output/` | 标定产生的图（距离分布 + PR 曲线） |
| `scripts/` | 工具脚本（拆分笔记、生成测试集） |
| `faiss_index/` | FAISS 索引（583 chunk，已提交进 Git） |
| `hf_cache/` | HuggingFace 模型缓存（embedding + reranker，volume 挂载） |
| `data/` | checkpoints.db 持久化目录（volume 挂载，不提交） |

## 启动

### 本地开发

```bash
# 1. 启动 MySQL + Redis
docker compose -f docker-compose.db.yml up -d

# 2. 启动 API 服务
cd rag_api
uvicorn api_agent:app --reload
```

访问 http://127.0.0.1:8000/docs

### Docker 部署

```bash
# 1. 首次使用前：本地下载模型到 hf_cache/
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='hf_cache/hub')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-base', cache_dir='hf_cache/hub')"

# 2. 构建并启动（首次约 5~10 分钟）
docker compose up --build -d

# 3. 后续启动（镜像已存在，几秒）
docker compose up -d

# 4. 查看日志
docker compose logs -f agent-api

# 5. 停止
docker compose down
```

### 开发期 volume 挂载

`docker-compose.yml` 里 `agent-api` 服务挂载了这些目录：

```yaml
volumes:
  - ./hf_cache:/root/.cache/huggingface
  - ./data:/app/data
  - ./rag_api:/app/rag_api
  - ./mcp_test:/app/mcp_test
  - ./rag_core:/app/rag_core
  - ./docs:/app/docs
  - ./faiss_index:/app/faiss_index
```

好处：**改代码不用 rebuild**。改完 `rag_api/api_agent.py` 只要：

```bash
docker compose restart agent-api
```

改 `docs/` 或 `faiss_index/` 也是 restart 就行，不用 build。

**收尾（生产冻结）时**：把 `rag_api`、`mcp_test`、`rag_core`、`docs`、`faiss_index` 这几个挂载注释掉，重新 `docker compose build`，镜像就是自包含的。

## 接口

| 方法 | 路径 | 鉴权 | 限流 | 说明 |
|---|---|---|---|---|
| GET | `/` | 否 | 否 | 健康检查 |
| POST | `/chat` | ✅ Bearer token | 20/分钟 | 普通对话 |
| POST | `/chat/stream` | ✅ Bearer token | 20/分钟 | 流式对话（SSE） |
| GET | `/history` | 否 | 否 | 查看对话历史（MySQL 展示层） |

### 鉴权格式

所有需要鉴权的接口，请求头里带：

```text
Authorization: Bearer <API_KEY>
```

- 不带 → 401 `{"detail": "缺少 Authorization 头"}`
- 格式不对 → 401 `{"detail": "Authorization 格式错误..."}`
- token 不对 → 403 `{"detail": "API Key 无效"}`
- token 正确 → 正常返回

### 限流格式

按 API Key 限流，默认 20 次/分钟。超限返回：

```json
{"detail": "请求过于频繁，每分钟最多 20 次，请稍后再试"}
```

限流配额通过环境变量 `RATE_LIMIT_PER_MINUTE` 配置。**Redis 不可用时放行**（fail-open），限流不会成为新的单点故障。

### 请求示例

```bash
# 普通对话（带鉴权 + thread_id 区分会话）
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{"message": "MCP 是什么？", "thread_id": "user_001"}'

# 流式对话
curl -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{"message": "MCP 是什么？", "thread_id": "user_001"}'

# 查看历史（不需要鉴权）
curl "http://127.0.0.1:8000/history?thread_id=user_001"
```

### 多轮对话演示

```bash
# 第一轮
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{"message": "MMR 是什么？", "thread_id": "test_001"}'

# 第二轮（用代词指代）
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{"message": "它和普通相似度检索有什么区别？", "thread_id": "test_001"}'

# 重启服务后再问，仍能理解"它"指 MMR
# docker compose down && docker compose up -d 后依然成立（checkpoints volume 持久化）
```

## 环境变量

复制 `.env.example` 为 `.env`，填写：

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
MYSQL_PASSWORD=your_mysql_password
API_KEY=your-secret-key-here
RATE_LIMIT_PER_MINUTE=20
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | — | DeepSeek API 地址 |
| `MYSQL_PASSWORD` | `root123` | MySQL 密码 |
| `API_KEY` | `dev-key-change-me` | 接口鉴权用的 Bearer token |
| `RATE_LIMIT_PER_MINUTE` | `20` | 每分钟限流配额（按 API Key 计） |

**Docker 容器额外环境变量**（由 docker-compose.yml 注入）：

| 变量 | 值 | 说明 |
|---|---|---|
| `MYSQL_HOST` | `mysql` | 容器内用服务名，不用 localhost |
| `REDIS_HOST` | `redis` | 容器内用服务名，本地开发用 `localhost` |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `HF_HOME` | `/root/.cache/huggingface` | 模型缓存目录 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像 |
| `HF_HUB_OFFLINE` | `1` | 离线模式 |
| `TRANSFORMERS_OFFLINE` | `1` | 离线模式 |
| `DATA_DIR` | `/app/data` | checkpoints 持久化目录 |
| `API_KEY` | `${API_KEY:-dev-key-change-me}` | 从宿主机 `.env` 读，读不到用默认值 |
| `RATE_LIMIT_PER_MINUTE` | `${RATE_LIMIT_PER_MINUTE:-20}` | 从宿主机 `.env` 读，读不到用默认值 |

## RAG 评估

构建 **249 条测试项**（覆盖 83 篇文档），5 种策略对比：

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 50.6% | 0.23 | 0.39 | 2.63 |
| MMR | 37.8% | 0.15 | 0.33 | 2.79 |
| MMR + 按来源去重 | 40.6% | 0.14 | 0.34 | **3.00** |
| 混合检索（向量+BM25+RRF） | 52.6% | 0.24 | 0.42 | 2.57 |
| **混合检索 + Rerank** | **57.0%** | **0.25** | **0.47** | 2.56 |
| **随机基线** | **3.0%** | — | — | — |

**关键发现**：
- **随机基线 3.0%** → 判别空间扩大到 **54.0 个点**
- **Rerank 增益显著**：召回率 +4.4%（52.6% → 57.0%），MRR +0.05
- **混合检索优于纯相似度**：召回率 52.6% vs 50.6%（+2.0%）
- **MMR 在"精确匹配"任务上吃亏**：召回率只有 37.8%（多样性导向，牺牲精度）
- **多样性 trade-off 明显**：MMR+去重 3.00 vs 相似度 2.63

> **可复现**：`python eval/eval_retrieval.py` 一条命令跑出上面所有数字。

## 测试集自动生成

`eval/test_set_v2.json` 里的 249 条测试项是通过脚本自动生成的：

**流程**：
1. 遍历所有 chunk，按 source 分组
2. 每篇文档挑 2 个最长 chunk
3. 调 DeepSeek API 生成 3 个问题
4. 标注 `expected_sources = 该文档的 source`

**脚本**：`python scripts/generate_testset.py`
**成本**：约 ¥0.2（98 篇 × 1 次 API 调用）

## 阈值标定

`MAX_DISTANCE` 降级阈值通过实验标定，不是拍脑袋定的。

**当前语料（583 chunk）的标定结果**：

| 阈值 | Precision | Recall | F1 | 业务含义 |
|---|---|---|---|---|
| 1.00（旧值） | 0.967 | 0.967 | 0.967 | 漏答 3.3%，幻觉 3.3% |
| **0.96（F1 最大）** | **0.990** | **1.000** | **0.995** | **漏答 0%，幻觉 1.0%** |

**距离分布**：
- 正样本：0.363 ~ 0.958（中位数 0.709）
- 负样本：0.813 ~ 1.420（中位数 1.176）
- 重叠区：0.813 ~ 0.958，宽 0.145

**复现**：`python eval/calibrate_distance.py`

## 技术栈

- **Agent 编排**：LangChain 1.4 + LangGraph 1.2
- **模型**：DeepSeek（OpenAI 兼容协议）
- **工具协议**：MCP（FastMCP 3.4.8）
- **Embedding**：bge-small-zh-v1.5（512 维，双塔）
- **Rerank**：bge-reranker-base（CrossEncoder）
- **检索**：sentence-transformers + FAISS + BM25 + 加权 RRF
- **鉴权**：Bearer token + FastAPI Depends
- **限流**：Redis 固定窗口（`incr + expire`）
- **降级**：向量距离阈值（标定 0.96）
- **记忆持久化**：AsyncSqliteSaver
- **展示层**：MySQL
- **服务**：FastAPI + Uvicorn
- **部署**：Docker + Docker Compose（多阶段构建，2.42 GB）
- **流式**：SSE

## 设计决策

| 决策 | 原因 |
|---|---|
| 记忆统一到 LangGraph checkpointer | 避免 InMemorySaver + MySQL 双源打架 |
| checkpoints 挂载到宿主机 | `docker compose down` 后记忆不丢 |
| 删掉 Redis 缓存 | 命中率结构性为 0，是负优化 |
| 检索层降级（距离阈值） | 比 Prompt 软约束更可靠 |
| 错误映射为友好提示 | `str(e)` 可能泄漏 API key、路径等 |
| 混合检索加权 RRF | 等权 RRF 实测反而比纯向量差 |
| 抽公共 `hybrid_search` 到 `rag_core/` | 服务端和评估脚本共用一份代码 |
| `MAX_DISTANCE` 标定为 0.96 | 通过距离分布 + PR 曲线找 F1 最大点 |
| **API Key 鉴权** | **用 Depends 依赖注入，鉴权逻辑和业务逻辑解耦** |
| **Redis 限流** | **按 API Key 限流，`incr + expire` 固定窗口，fail-open** |
| **开发期 volume 挂载** | **改代码不用 rebuild，迭代秒级** |
| Docker 多阶段构建 | 镜像从 3.51 GB 降到 2.42 GB |
| CPU 版 torch | 下载量从 5 GB 降到 600 MB |

## 已知不足

- pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环（生产应换 `aiomysql`）
- `save_conversation` 每次请求新建 MySQL 连接，无连接池
- 评估是**文档级召回**而非 chunk 级——source 级指标高估了 8~14 个点
- 语料里混着"面试题标准答案"，**存在语料同源问题**
- 流式状态机用布尔变量表达，边界情况可能出错
- **Rerank 单次推理 50~100 ms（CPU）**，249 条查询需 4~8 分钟；高并发场景需要优化

## 学习路径

ReAct → LangChain → LangGraph → RAG → MCP → FastAPI → Docker → 鉴权 → 限流 → 开发期 volume 挂载