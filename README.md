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
                    rag_core/search.py（公共混合检索）
                            ↓
                    降级判断（距离阈值 0.96）
                            ↓ 通过
                    混合检索召回 Top-20（向量 + BM25 + 加权 RRF）
                            ↓
                    Rerank 精排 Top-3（bge-reranker-base, CrossEncoder）
                            ↓
                    FAISS 向量库（98 篇文档 / 576 chunk）
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

- **持久化记忆**：LangGraph checkpointer + AsyncSqliteSaver，服务重启/容器重建不丢历史
- **RAG 检索增强**：本地 embedding（BAAI/bge-small-zh-v1.5）+ FAISS 持久化
- **两阶段检索**：混合检索召回 Top-20 → Rerank 精排 Top-3
- **混合检索**：向量 + BM25 + 加权 RRF（向量 0.7 / BM25 0.3）
- **Rerank 精排**：bge-reranker-base（CrossEncoder），召回率 +6.0%，MRR +0.04
- **公共检索模块**：`rag_core/search.py`，服务端和评估脚本共用同一份 `hybrid_search`
- **降级策略**：向量距离阈值（标定为 0.96），检索不到就不调 LLM，避免幻觉
- **MCP 工具解耦**：工具独立成 Server，跨进程 stdio 通信
- **SSE 流式输出**：逐 token 返回，中间步骤过滤
- **MySQL 展示层**：对话历史按 thread_id 持久化，供 `/history` 接口查询
- **Agent 护栏**：`recursion_limit=10` + `max_tokens=2000`，防止死循环和 Token 爆炸
- **友好错误处理**：异常映射为可读提示，不泄漏 API key、路径等内部信息
- **RAG 评估**：249 条测试项，5 种策略对比 + 随机基线（3.1%），一条命令可复现
- **测试集自动生成**：脚本用 DeepSeek API 批量生成 200+ 条
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
| `docs/notes/` | 项目笔记（16 篇） |
| `docs/notes_interview/` | 面试题逐个拆分（25 篇） |
| `docs/readme/` | README 按章节拆分（13 篇） |
| `docs/stories/` | 踩坑故事（33 篇） |
| `eval/` | RAG 评估脚本 + 测试集（249 条）+ 阈值标定 |
| `eval/output/` | 标定产生的图（距离分布 + PR 曲线） |
| `scripts/` | 工具脚本（拆分笔记、生成测试集） |
| `faiss_index/` | FAISS 索引（576 chunk，已提交进 Git） |
| `hf_cache/` | HuggingFace 模型缓存（embedding + reranker，volume 挂载） |
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
# 1. 首次使用前：本地下载模型到 hf_cache/
#    embedding 模型 192 MB + reranker 模型 3.2 GB
#    注意：必须指定 cache_dir，否则默认写到 ~/.cache/huggingface，
#    容器 volume 挂载的 ./hf_cache/ 会是空的

# 1a. embedding 模型
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='hf_cache/hub')"

# 1b. reranker 模型
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
# docker compose down && docker compose up -d 后依然成立（checkpoints volume 持久化）
```

## 环境变量

复制 `.env.example` 为 `.env`，填写：

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
MYSQL_PASSWORD=your_mysql_password
```

**Docker 容器额外环境变量**（由 docker-compose.yml 注入）：

| 变量 | 值 | 说明 |
|---|---|---|
| `MYSQL_HOST` | `mysql` | 容器内用服务名，不用 localhost |
| `HF_HOME` | `/root/.cache/huggingface` | 模型缓存目录 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像 |
| `HF_HUB_OFFLINE` | `1` | 离线模式 |
| `TRANSFORMERS_OFFLINE` | `1` | 离线模式 |
| `DATA_DIR` | `/app/rag_api/data` | checkpoints 持久化目录 |

## RAG 评估

构建 **249 条测试项**（覆盖 83 篇文档），5 种策略对比：

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 52.2% | 0.24 | 0.40 | 2.63 |
| MMR | 41.0% | 0.16 | 0.35 | 2.79 |
| MMR + 按来源去重 | 44.2% | 0.15 | 0.36 | **3.00** |
| 混合检索（向量+BM25+RRF） | 54.6% | 0.24 | 0.45 | 2.60 |
| **混合检索 + Rerank** | **60.6%** | **0.27** | **0.49** | 2.60 |
| **随机基线** | **3.1%** | — | — | — |

**关键发现**：
- **随机基线 3.1%** → 判别空间扩大到 **57.5 个点**（旧测试集只有 15.9 个点）
- **Rerank 增益显著**：召回率 +6.0%（54.6% → 60.6%），MRR +0.04
- **混合检索优于纯相似度**：召回率 54.6% vs 52.2%（+2.4%）
- **MMR 在"精确匹配"任务上吃亏**：召回率只有 41%（多样性导向，牺牲精度）
- **多样性 trade-off 明显**：MMR+去重 3.00 vs 相似度 2.63

> **可复现**：`python eval/eval_retrieval.py` 一条命令跑出上面所有数字。首次跑会加载 Reranker 模型（10~20 秒）。

## 测试集自动生成

`eval/test_set_v2.json` 里的 249 条测试项是通过脚本自动生成的：

**流程**：
1. 遍历所有 chunk，按 source 分组
2. 每篇文档挑 2 个最长 chunk
3. 调 DeepSeek API 生成 3 个问题（覆盖这 2 个 chunk 的内容）
4. 标注 `expected_sources = 该文档的 source`

**脚本**：`python scripts/generate_testset.py`

**成本**：约 ¥0.2（98 篇 × 1 次 API 调用）

**质量**：抽查前 10 条，问题口语化、角度多样、能被对应文档回答

## 阈值标定

`MAX_DISTANCE` 降级阈值通过实验标定，不是拍脑袋定的。

**方法**：
- 正样本 100 条（从 `test_set_v2.json` 抽样，seed=42 可复现）+ 负样本 25 条
- 对每条问题算 `best_distance`（top-1 的 L2 距离）
- 画正负样本距离分布直方图，找分离点
- 扫 101 个候选阈值（0.5~1.5），计算 Precision / Recall / F1

**当前语料（576 chunk）的标定结果**：

| 阈值 | Precision | Recall | F1 | 业务含义 |
|---|---|---|---|---|
| 1.00（旧值） | 0.980 | 1.000 | 0.990 | 漏答 0%，幻觉 2.0% |
| **0.96（F1 最大）** | **0.990** | **1.000** | **0.995** | **漏答 0%，幻觉 1.0%** |

**距离分布**：
- 正样本：0.363 ~ 0.958（中位数 0.708）
- 负样本：0.813 ~ 1.420（中位数 1.176）
- 重叠区：0.813 ~ 0.958，宽 0.145

**为什么选 0.96**：F1 最大点，精确率 99.0%（幻觉率仅 1.0%），召回率保持 100%。相比 1.0 又多挤出 1% 精确率。

**复现**：

```bash
python eval/calibrate_distance.py
```

输出：
- `eval/output/distance_distribution.png`（距离分布直方图）
- `eval/output/pr_curve.png`（阈值扫描曲线）

## 技术栈

- **Agent 编排**：LangChain 1.4 + LangGraph 1.2
- **模型**：DeepSeek（OpenAI 兼容协议）
- **工具协议**：MCP（FastMCP 3.4.8）
- **Embedding**：bge-small-zh-v1.5（512 维，双塔）
- **Rerank**：bge-reranker-base（CrossEncoder，交叉编码器）
- **检索**：sentence-transformers + FAISS + BM25 + 加权 RRF（`rag_core/search.py` 公共模块）
- **降级**：向量距离阈值（标定 0.96）
- **记忆持久化**：AsyncSqliteSaver（生产可换 PostgresSaver）
- **展示层**：MySQL
- **服务**：FastAPI + Uvicorn
- **部署**：Docker + Docker Compose（多阶段构建，2.42 GB）
- **流式**：SSE

## 设计决策

| 决策 | 原因 |
|---|---|
| 记忆统一到 LangGraph checkpointer | 避免 InMemorySaver + MySQL 双源打架 |
| checkpoints 挂载到宿主机 | `docker compose down` 后记忆不丢（容器重建也持久） |
| 删掉 Redis 缓存 | 命中率结构性为 0，是负优化 |
| 检索层降级（距离阈值） | 比 Prompt 软约束更可靠，从根源避免幻觉 |
| 错误映射为友好提示 | `str(e)` 可能泄漏 API key、路径等内部信息 |
| 混合检索加权 RRF | 等权 RRF 实测反而比纯向量差 |
| 抽公共 `hybrid_search` 到 `rag_core/` | 服务端和评估脚本共用一份代码，保证"评估跑的就是部署的" |
| `MAX_DISTANCE` 标定为 0.96 | 通过正负样本距离分布 + PR 曲线找到 F1 最大点（P=0.990, R=1.000） |
| **两阶段检索（召回 + Rerank）** | **粗排快（双塔）+ 精排准（交叉编码器），召回率 +6.0%** |
| **语料 98 篇 / 576 chunk** | **扩大判别空间，避免天花板效应** |
| **测试集 249 条（自动生成）** | **统计意义充分，随机基线降到 3.1%** |
| **拆分 `notes/07` 为 25 个独立文件** | **提升文档级召回判别力（原一个文件 74 chunk）** |
| Docker 多阶段构建 | builder 阶段装编译器，运行时不带，镜像从 3.51 GB 降到 2.42 GB |
| CPU 版 torch | 项目只用 bge-small 做 embedding，CPU 毫秒级；下载量从 5 GB 降到 600 MB |
| 模型缓存 volume 挂载 | 模型下载移出构建期，避免网络超时；重建不重下 |
| HF_HOME 三级探测 | 环境变量 > 容器默认挂载点 > 项目内 hf_cache，本地和容器都能跑 |
| FAISS 索引提交进 Git | 576 chunk 索引约 1~2 MB，别人 clone 后可直接构建 |

## 已知不足

- pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环（生产应换 `aiomysql`）
- `save_conversation` 每次请求新建 MySQL 连接，无连接池
- 评估是**文档级召回**而非 chunk 级——source 级指标高估了 8~14 个点（实测 source 召回 60.6% vs chunk 召回约 52%）
- 语料里混着"面试题标准答案"，**存在语料同源问题**——评估结论不能直接外推到真实用户场景
- 流式状态机用布尔变量表达，边界情况可能出错（P1-A 待做）
- **Rerank 单次推理 50~100 ms（CPU）**，249 条查询需 4~8 分钟；高并发场景需要优化（更小模型 / GPU / 缓存）

## 学习路径

ReAct → LangChain → LangGraph → RAG → MCP → FastAPI → Docker