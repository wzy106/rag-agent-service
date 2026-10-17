# RAG Agent 服务

基于 LangChain + LangGraph + MCP + RAG 的 Agent 服务，支持 SSE 流式输出、多轮记忆持久化、API Key 多租户鉴权、Redis 限流、Docker 部署，并自带一套可复现的评估体系（267 条 v3 测试集：evidence chunk + 多金标 + source/chunk 双粒度 + 端到端 LLM-as-judge）。

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
                    混合检索召回 Top-N（向量 + BM25 + 加权 RRF，默认 N=20 可配）
                            ↓
                    Rerank 精排 Top-3（bge-reranker-base, CrossEncoder）
                            ↓
                    FAISS 向量库（98 篇文档 / 617 chunk）
                            ↓ 检索结果返回给 Agent
                        DeepSeek 生成回答
    ↓ SSE 流式返回
客户端

鉴权层：
- API Key 通过 Bearer token 传入
- Header + Depends 做依赖注入，鉴权逻辑和业务逻辑解耦
- API_KEY / API_KEYS 从环境变量读，不写死在代码里
- 认证 ≠ 授权：`verify_key` 只负责认证并返回租户 id（principal），具体能读到哪些数据由各接口按 principal 过滤

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
- **API Key 鉴权**：`/chat`、`/chat/stream`、`/history` 都需要 Bearer token（仅 `/` 健康检查公开）；支持多 key，每个 key 一个租户，会话与历史互相隔离
- **Redis 限流**：按 API Key 限流，20 次/分钟（`incr + expire` 固定窗口，fail-open）
- **RAG 检索增强**：本地 embedding（BAAI/bge-small-zh-v1.5）+ FAISS 持久化
- **两阶段检索**：混合检索召回 Top-N（`RAG_RECALL_K`，默认 20）→ Rerank 精排 Top-3
- **混合检索**：向量 + BM25 + 加权 RRF（向量 0.7 / BM25 0.3）
- **Rerank 精排**：bge-reranker-base（CrossEncoder）；效果按 candidate recall@20 / final recall@3 分阶段评估（见「RAG 评估」）
- **公共检索模块**：`rag_core/search.py`，服务端和评估脚本共用同一份 `hybrid_search`
- **降级策略**：向量距离阈值（0.96）；注意它只 gate 检索，不 gate 生成（见「已知不足」）
- **MCP 工具解耦**：工具独立成 Server，跨进程 stdio 通信；**常驻 session**（`client.session()` + `load_mcp_tools`，子进程只拉起一次，模型不重复加载）
- **检索可配置**：候选数 `RAG_RECALL_K`（默认 20）、精排截断 `RAG_RERANK_MAX_LENGTH`、精排开关 `RAG_ENABLE_RERANK`
- **SSE 流式输出**：逐 token 返回，中间步骤过滤
- **MySQL 展示层**：对话历史按「租户 + thread_id」持久化，`/history` 只能查到自己租户的记录
- **Agent 护栏**：`recursion_limit=10` + `max_tokens=2000`，防止死循环和 Token 爆炸
- **友好错误处理**：异常映射为可读提示，不泄漏 API key、路径等内部信息
- **RAG 评估**：267 条 v3 测试项（evidence chunk + 多金标），5 种策略对比 + candidate/final 分阶段指标 + 随机基线 + Wilson/bootstrap 置信区间
- **测试集自动生成**：脚本用 DeepSeek API 批量生成带 evidence chunk 的问题
- **阈值标定**：easy/hard 负样本距离分布 + PR 曲线，分别报告 FPR / FDR
- **端到端评估**：LLM-as-judge 评 faithfulness / relevance / refusal，并输出「检索命中 × 回答忠实」2×2 交叉表
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
| `eval/` | RAG 评估脚本 + v3 测试集 + 多金标 + 阈值标定 + 端到端评估 |
| `eval/output/` | 检索/标定/端到端评估结果（JSON + 图） |
| `scripts/` | 工具脚本（拆分笔记、生成测试集、多金标、建索引） |
| `tests/` | pytest 测试（单元 + 集成） |
| `.github/workflows/` | GitHub Actions CI |
| `faiss_index/` | FAISS 索引（617 chunk，已提交进 Git） |
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
| POST | `/chat` | ✅ Bearer token | 20/分钟 | 普通对话（thread_id 必填） |
| POST | `/chat/stream` | ✅ Bearer token | 20/分钟 | 流式对话（SSE） |
| GET | `/history` | ✅ Bearer token | 否 | 查看对话历史（只返回**自己租户**的记录） |

### 鉴权格式

所有需要鉴权的接口，请求头里带：

```text
Authorization: Bearer <API_KEY>
```

- 不带 → 401 `{"detail": "缺少 Authorization 头"}`
- 格式不对 → 401 `{"detail": "Authorization 格式错误..."}`
- token 不对 → 403 `{"detail": "API Key 无效"}`
- token 正确 → 正常返回

**多租户**：`API_KEYS` 支持逗号分隔的多个 key，**每个 key 是一个独立租户**：

- 鉴权通过后返回的是该 key 对应的租户 id（`sha256(key)[:16]`），**不是 key 本身**
- 记忆（LangGraph checkpointer）按 `"{租户}:{thread_id}"` 命名空间隔离——同一个 `thread_id` 在不同 key 下互不影响
- 对话历史（MySQL）带 `owner` 列，`/history` 只返回 `owner = 当前租户` 的记录

> 只有单个 `API_KEY` 时，所有持有者同属一个租户——真隔离需要一 key 一人。

### 限流格式

按租户（API Key）限流，默认 20 次/分钟。超限返回：

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

# 查看历史（需要鉴权；只返回自己租户的该 thread_id 的记录）
curl "http://127.0.0.1:8000/history?thread_id=user_001" \
  -H "Authorization: Bearer your-secret-key-here"
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
| `ANSWER_MODEL` | `deepseek-chat` | 端到端评估的回答模型（可选） |
| `JUDGE_MODEL` | `deepseek-chat` | LLM-as-judge 模型（可选，换模型可降低自评偏好） |
| `JUDGE_API_KEY` | 复用 `DEEPSEEK_API_KEY` | judge 的 API key（可选） |
| `JUDGE_BASE_URL` | 复用 `DEEPSEEK_BASE_URL` | judge 的 base URL（可选） |
| `MYSQL_PASSWORD` | `root123` | MySQL 密码 |
| `API_KEY` | `dev-key-change-me` | 接口鉴权用的 Bearer token（单租户） |
| `API_KEYS` | — | 逗号分隔的多个 key，每个 key 一个独立租户；设置后优先于 `API_KEY` |
| `RATE_LIMIT_PER_MINUTE` | `20` | 每分钟限流配额（按租户计） |
| `RAG_RECALL_K` | `20` | 混合检索粗排召回候选数（越大越准越慢） |
| `RAG_RERANK_MAX_LENGTH` | `512` | Reranker 输入截断长度（token） |
| `RAG_ENABLE_RERANK` | `1` | 置 `0` 关闭精排，直接返回粗排 Top-3 |

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
| `API_KEYS` | `${API_KEYS:-}` | 多租户 key 列表，从宿主机 `.env` 读 |
| `RATE_LIMIT_PER_MINUTE` | `${RATE_LIMIT_PER_MINUTE:-20}` | 从宿主机 `.env` 读，读不到用默认值 |

## 测试

```bash
pip install -r requirements-dev.txt

pytest -m "not integration"   # 单元测试：快，不需要模型
pytest -m integration         # 集成测试：需要本地 HF 模型 + FAISS 索引
pytest                        # 全部
```

| 层级 | 覆盖内容 | 需要模型 |
|---|---|---|
| 单元（`test_auth.py` / `test_api_units.py`） | 鉴权、租户隔离、`/history` 越权、`thread_id` 校验、错误映射、限流（含 fail-open） | 否 |
| 单元（`test_chunking.py` / `test_eval_metrics.py` / `test_hybrid_search_units.py`） | chunk_id/section_path、重复内容不塌缩、召回/MRR/nDCG/CI | 否 |
| 单元（`test_index_consistency.py`） | 提交的 FAISS 索引与 docs 重新切分后的 chunk_id 集合一致 | 否 |
| 集成（`test_search_integration.py`） | 索引加载、元数据、`hybrid_search` 返回数、距离阈值降级、精排开关 | 是 |

- 接口测试用**打桩**（假 Agent / 假 MySQL / 假 Redis），不依赖真实 LLM 或数据库
- **CI**：`.github/workflows/ci.yml`，push / PR 时跑单元测试（集成测试需下载模型，CI 跳过）
- `rag_core.search` 对 sentence-transformers 做懒加载，没装 torch 时也能导入跑纯函数测试
- 测试用的 API key 固定为 `key_alice` / `key_bob`，不读真实 `.env`

## RAG 评估

**267 条 v3 测试项**（覆盖 89 篇文档 / 617 chunk），5 种策略对比。数字来自 `eval/output/retrieval_v3.json`：

| 策略 | source strict | source relaxed | chunk any | chunk all | candidate@20 | precision | MRR | nDCG@3 | diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 普通相似度 | 62.9% | 63.7% | 47.2% | 41.9% | n/a | 0.167 | 0.341 | 0.357 | 2.61 |
| MMR | 50.2% | 50.9% | 30.3% | 24.3% | n/a | 0.101 | 0.265 | 0.253 | 2.84 |
| MMR + 按来源去重 | 52.4% | 53.2% | 30.3% | 24.3% | n/a | 0.101 | 0.264 | 0.252 | **3.00** |
| 混合检索（向量+BM25+RRF） | 64.4% | 65.9% | 51.3% | 46.4% | n/a | 0.181 | 0.389 | 0.404 | 2.57 |
| **混合检索 + Rerank** | **69.3%** | **70.0%** | **59.9%** | **53.9%** | **77.9%** | **0.211** | **0.498** | **0.502** | 2.53 |
| **随机基线** | **3.3%** | **3.8%** | **0.6%** | — | — | — | — | — | — |

**关键发现**：
- **source 级比 chunk 级高 9.4 个点**（Rerank：69.3% vs 59.9%）——只报 source 会高估真实能力。
- **候选阶段比最终阶段高 18.0 个点**（candidate@20 77.9% vs final@3 59.9%）；Top-20 里有正确块，但精排 Top-3 只留下 59.9%，最终截断是主要瓶颈。
- **Rerank vs 混合检索**：chunk any **+8.6%**（95% CI +3.4% ~ +14.2%，p=0.004）；source strict **+4.9%**（95% CI -0.4% ~ +10.1%，p=0.084，**不显著**）。同一个改动，不同粒度给出不同结论。
- **混合检索优于纯相似度**：source 64.4% vs 62.9%（+1.5%），chunk 51.3% vs 47.2%（+4.1%）。
- **MMR 在精确匹配任务上吃亏**：chunk any 30.3%，低于相似度 47.2%；但多样性更高（2.84）。
- **多金标确实有影响**：source relaxed 比 strict 高 0.7–1.5 个点；v3 中 11/267（4.1%）的样本在别的 source 有 ≥0.8 的近重复可答块。

> **可复现**：
> ```bash
> python eval/eval_retrieval.py \
>   --test-set eval/test_set_v3_gold.json --k 3 --bootstrap 2000 \
>   --json eval/output/retrieval_v3.json
> ```

### 端到端评估（80 正 + 20 负）

`eval/eval_e2e.py` 跑“检索 + 生成 + LLM-as-judge”，结果在 `eval/output/e2e_summary.json`：

| 指标 | 值 | 95% CI |
|---|---:|---|
| 可回答问题 n | 80 | — |
| 实际作答 | 66 | — |
| 有效拒答（资料不足） | 14（17.5%） | — |
| faithfulness（作答项） | **100.0%**（66/66） | 94.5% ~ 100% |
| relevance（作答项） | **98.5%**（65/66） | 91.9% ~ 99.7% |
| unsupported claim rate | 0.0%（0/66） | 0% ~ 5.5% |
| 负样本拒答准确率 | **100.0%**（20/20） | 83.9% ~ 100% |

**2×2 交叉表（检索命中 × 回答忠实）**：

| | faithful=True | faithful=False |
|---|---:|---:|
| retrieval_hit=True | 37 | 0 |
| retrieval_hit=False | 29 | 0 |

**怎么读这组数**：
- 在这个自建语料 + 自生成问题的设定里，生成层几乎不编造；**主要瓶颈是检索**：29/66 的回答是在检索未命中 gold 的情况下生成的，另外 14/80 因资料不足拒答。
- `retrieval_hit=False, faithful=True` 不等于回答正确：可能是命中了近重复/等价 chunk，而 gold 标签没覆盖；也可能回答只对上下文忠实、但没解决用户问题。faithfulness ≠ correctness。
- 100% 是上界，不是普适结论：judge 默认与回答模型同为 DeepSeek，存在自评偏好；需要换 judge 模型并人工抽查。当前只人工抽查了少量样本。
- 端到端评估是直接 RAG 链路，不包含 MCP 工具选择、多轮改写和 Agent 循环。

**延迟（本机 CPU）**：检索 P50 6.6s / P95 7.8s；生成 P50 0.9s；judge P50 0.9s。检索是主要延迟来源（Reranker）。

> **可复现**：
> ```bash
> python eval/eval_e2e.py \
>   --test-set eval/test_set_v3_gold.json \
>   --negatives eval/negatives.json eval/negatives_hard.json \
>   --answerable 80 --negative 20 --seed 42 --concurrency 3
> ```

## 测试集自动生成

`eval/test_set_v3.json` 里的 267 条测试项由脚本自动生成，v2 保留用于历史对比：

**流程**：
1. 遍历索引 chunk，按 source 分组
2. 每篇文档挑 2 个最长 chunk
3. 调 DeepSeek API 生成 3 个问题，并让模型标注 evidence（哪些 chunk 真正包含答案）
4. 写入 `expected_sources` + `evidence_chunk_ids` + `evidence_coverage_mode`
5. `scripts/label_gold.py` 用 3-gram Jaccard ≥ 0.8 扩展近重复块，生成 `gold_chunk_ids` / `gold_sources`

**脚本**：

```bash
python scripts/generate_testset.py --output eval/test_set_v3.json
python scripts/label_gold.py --test-set eval/test_set_v3.json
```

**成本**：约 ¥0.2–0.5（98 篇 × 1 次 API 调用）

## 阈值标定

`MAX_DISTANCE` 降级阈值通过实验标定，不是拍脑袋定的。但它衡量的是**检索层是否接受 query**，不是端到端幻觉率。

**当前语料（617 chunk，100 正样本 + easy/hard 负样本）**：

| 负样本组 | n | FPR @ 0.96 | 说明 |
|---|---:|---:|---|
| easy（天气/菜谱/电影） | 25 | **4.0%**（1/25） | 明显离题 |
| hard（版本号/新特性/价格） | 20 | **95.0%**（19/20） | 近域但库内没有 |
| combined | 45 | **44.4%**（20/45） | — |

在 100 正 + 45 负上：**Precision 0.833、Recall 1.000、F1 0.909**。

F1 最大点是 **0.87**（Precision 0.884、Recall 0.990、F1 0.934），但 hard FPR 仍有 **60%**。这说明**单一距离阈值无法区分“近域但库内没有”的问题**；0.96 保留用于拦截明显离题 query，真正的拒答需要生成层的充分性判断或 graph 分支。

**口径**：
- `FPR = FP/(FP+TN)`：负样本被接受的比例
- `FDR = FP/(TP+FP)`：被接受结果里负样本占的比例，依赖抽样基率
- `1 - precision` 不是“幻觉率”；旧文档的“幻觉率 1%”说法已修正

**复现**：

```bash
python eval/calibrate_distance.py \
  --positives eval/test_set_v3_gold.json \
  --negatives eval/negatives.json eval/negatives_hard.json \
  --output eval/output/calibration_v3.json
```

## 技术栈

- **Agent 编排**：LangChain 1.4 + LangGraph 1.2
- **模型**：DeepSeek（OpenAI 兼容协议）
- **工具协议**：MCP（FastMCP 3.4.8）
- **Embedding**：bge-small-zh-v1.5（512 维，双塔）
- **Rerank**：bge-reranker-base（CrossEncoder）
- **检索**：sentence-transformers + FAISS + BM25 + 加权 RRF
- **鉴权**：Bearer token + FastAPI Depends（多 key 多租户隔离）
- **限流**：Redis 固定窗口（`incr + expire`）
- **测试**：pytest（单元 + 集成）+ GitHub Actions CI
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
| `MAX_DISTANCE` 标定为 0.96 | 通过距离分布 + PR 曲线；只用于拦截明显离题 query，hard 负样本仍会通过 |
| **chunk_id + section_path** | **稳定 ID 用 source + section + content hash，避免重复内容互相覆盖** |
| **source/chunk 双粒度 + 多金标** | **source 任意块命中会高估，单金标会低估；近重复扩展为 gold_chunk_ids/gold_sources** |
| **端到端 LLM-as-judge** | **检索命中不代表回答忠实；用 2×2 交叉表区分检索问题和生成问题** |
| **评估脚本拆 `main()`** | **可被 pytest 导入，加纯函数单测和索引一致性测试** |
| **API Key 鉴权** | **用 Depends 依赖注入，鉴权逻辑和业务逻辑解耦** |
| **Redis 限流** | **按 API Key 限流，`incr + expire` 固定窗口，fail-open** |
| **开发期 volume 挂载** | **改代码不用 rebuild，迭代秒级** |
| Docker 多阶段构建 | 镜像从 3.51 GB 降到 2.42 GB |
| CPU 版 torch | 下载量从 5 GB 降到 600 MB |

## 已知不足

- 语料是项目自述（98 篇笔记/面试题/踩坑故事），测试集也由这些 chunk 生成；v3 的 evidence chunk + 多金标只能修正“同一答案在别的文档”，不能消除“自己考自己”的外部效度问题。
- source 级和 chunk 级要同时报告：source 任意块命中会高估，单金标会低估；两者都不能单独代表真实能力。
- 阈值 `MAX_DISTANCE=0.96` 是检索层接受率，不是端到端幻觉率；实测 easy 负样本 FPR 4%、hard 负样本 FPR 95%。
- 阈值只 gate 检索，不 gate 生成：检索为空时工具返回“没有找到”，Agent 仍可能继续生成；真正拒答需要 graph 分支。
- 阈值只看向量 top-1，且在 BM25/RRF 之前判断；关键词强相关但向量距离大的 query 会被提前拒绝。
- 端到端评估是直接 RAG 链路（检索 + 生成 + judge），不覆盖 MCP 工具选择、多轮改写和 Agent 循环。
- LLM-as-judge 默认与回答模型同为 DeepSeek，存在自评偏好；可通过 `JUDGE_*` 环境变量换模型，并人工抽查校准。
- pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环（生产应换 `aiomysql`）
- `save_conversation` 每次请求新建 MySQL 连接，无连接池
- 流式状态机用布尔变量表达，边界情况可能出错，且没有自动化测试覆盖
- 依赖只有下界、没有 lockfile；干净环境复现仍需验证
- **Rerank 单次精排 CPU 上约 2~7 秒**（20 候选，本机实测，负载高时可达十几秒）
- **测试未覆盖 Agent 编排与流式**：`create_agent` 主流程和 SSE 状态机没有自动化测试

## 学习路径

ReAct → LangChain → LangGraph → RAG → MCP → FastAPI → Docker → 鉴权 → 限流 → 开发期 volume 挂载